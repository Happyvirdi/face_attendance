import cv2
import mediapipe as mp
import numpy as np
from utils import (
    load_all_face_embeddings,
    recognize_face,
    mark_attendance,
    is_person_marked_today,
    get_registered_faces
)
import time
import msvcrt
from collections import deque


class FaceRecognitionAttendance:
    def __init__(self, similarity_threshold=0.98):
        """
        Initialize face recognition system.
        
        Args:
            similarity_threshold: float, cosine similarity threshold (0.98 recommended)
        """
        self.similarity_threshold = similarity_threshold
        # Stricter threshold for actually marking attendance
        self.mark_similarity_threshold = 0.97
        # Tunable gating parameters
        self.margin_threshold = 0.03
        self.bbox_center_std_threshold = 8.0
        self.bbox_area_cv_threshold = 0.10
        
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Require more consecutive matches before marking attendance
        self.consecutive_match_required = 8
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Load registered faces
        self.registered_embeddings = load_all_face_embeddings()
    
    def get_face_embedding(self, frame):
        """
        Extract a shape-based face embedding from a frame.
        
        Args:
            frame: numpy array, video frame
        
        Returns:
            numpy array: discriminative face embedding or None
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # Convert to numpy [N,3]
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (468, 3)
            # Normalize by landmark bbox to reduce camera/pose effects
            min_xy = np.min(pts[:, :2], axis=0)
            max_xy = np.max(pts[:, :2], axis=0)
            center = (min_xy + max_xy) / 2.0
            scale = np.maximum(max_xy - min_xy, 1e-6)
            pts_xy_norm = (pts[:, :2] - center) / scale  # normalize x,y
            # Normalize z by its own std
            z = pts[:, 2]
            z_std = np.std(z) if np.std(z) > 1e-6 else 1.0
            z_norm = (z - np.mean(z)) / z_std
            # Downsample landmarks to reduce redundancy
            idx = np.arange(0, pts.shape[0], 16)  # ~30 points
            xy_sel = pts_xy_norm[idx, :]  # (K, 2)
            z_sel = z_norm[idx]           # (K,)
            # Radial histogram as a compact shape descriptor
            radii = np.sqrt(np.sum(xy_sel**2, axis=1))
            hist, _ = np.histogram(radii, bins=8, range=(0.0, np.max(radii) + 1e-6))
            hist = hist.astype(np.float32)
            if np.sum(hist) > 0:
                hist /= np.sum(hist)
            # Final embedding: flattened normalized coords + z + radial hist
            emb = np.concatenate([xy_sel.flatten(), z_sel.flatten(), hist], dtype=np.float32)
            return emb
        
        return None
    
    def has_face_detected(self, frame):
        """
        Check if a face is detected and get bounding box.
        
        Args:
            frame: numpy array, video frame
        
        Returns:
            tuple: (has_face: bool, bounding_box: list or None)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if results.detections:
            detection = results.detections[0]
            h, w, c = frame.shape
            
            bbox = detection.location_data.relative_bounding_box
            x_min = max(0, int(bbox.xmin * w) - 10)
            y_min = max(0, int(bbox.ymin * h) - 10)
            x_max = min(w, int((bbox.xmin + bbox.width) * w) + 10)
            y_max = min(h, int((bbox.ymin + bbox.height) * h) + 10)
            
            return True, [x_min, y_min, x_max, y_max]
        
        return False, None
    
    def draw_bounding_box_with_label(self, frame, bbox, label="", confidence=0.0):
        """
        Draw bounding box and label on frame.
        
        Args:
            frame: numpy array, video frame
            bbox: list, [x_min, y_min, x_max, y_max]
            label: string, person's name
            confidence: float, confidence score
        
        Returns:
            numpy array: frame with drawn elements
        """
        if not bbox:
            return frame
        
        x_min, y_min, x_max, y_max = bbox
        
        # Choose color based on recognition
        if label:
            color = (0, 255, 0)  # Green for recognized
        else:
            color = (0, 165, 255)  # Orange for unknown
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        if label:
            text = f"{label} ({confidence:.2f})"
            cv2.rectangle(frame, (x_min, y_min - 30), (x_max, y_min), color, -1)
            cv2.putText(frame, text, (x_min + 5, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (x_min, y_min - 30), (x_max, y_min), color, -1)
            cv2.putText(frame, f"Unknown ({confidence:.2f})", (x_min + 5, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def start_recognition(self):
        """
        Start real-time face recognition and attendance marking.
        """
        # Check if any faces are registered
        if not self.registered_embeddings:
            print("\n" + "="*60)
            print("ERROR: No registered faces found!")
            print("Please run 'python register_face.py' first to register faces.")
            print("="*60 + "\n")
            return
        
        # If only one face is registered, tighten thresholds further to avoid false positives
        registered_names = get_registered_faces()
        if len(registered_names) == 1:
            print("\nStrict mode: Only one registered face detected. Thresholds tightened.")
            self.similarity_threshold = max(self.similarity_threshold, 0.995)
            self.mark_similarity_threshold = max(self.mark_similarity_threshold, 0.998)
            self.consecutive_match_required = max(self.consecutive_match_required, 15)
        
        print("\n" + "="*60)
        print("FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*60)
        print(f"Registered faces: {', '.join(registered_names)}")
        print("\nInstructions:")
        print("  - Position your face in front of the camera")
        print("  - Attendance will be marked automatically")
        print("  - Press 'q' to quit")
        print("="*60 + "\n")
        
        # Create named window to ensure key events are captured and allow close-to-exit
        window_name = 'Face Recognition Attendance System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam!")
            return
        
        # Auto-stop configuration: stop after 1 minute (60 seconds)
        start_time = time.time()
        max_duration_sec = 60  # 1 minute
        
        last_marked_person = None
        last_marked_time = 0
        recognition_delay = 3  # Delay before allowing re-recognition
        
        frame_count = 0
        recognition_count = 0
        
        # Track consecutive confirmations for a recognized identity
        recognized_streak_name = None
        recognized_streak_count = 0
        streak_similarities = deque(maxlen=self.consecutive_match_required)
        bbox_history = deque(maxlen=self.consecutive_match_required)
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read from webcam!")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror frame
            h, w, c = frame.shape
            
            # Check for face
            has_face, bbox = self.has_face_detected(frame)
            
            current_time = time.time()
            recognized_name = None
            confidence = 0.0
            # Debug defaults
            best_score = None
            second_best_score = None
            margin_ok = False
            
            # Auto-stop after the configured duration
            if current_time - start_time >= max_duration_sec:
                print("\nAuto-stopping after 1 minute...")
                break
            
            if has_face:
                # Get embedding
                embedding = self.get_face_embedding(frame)
                
                if embedding is not None:
                    # Recognize face
                    recognized_name, confidence = recognize_face(
                        embedding,
                        self.registered_embeddings,
                        threshold=self.similarity_threshold
                    )
                    
                    # Compute similarity margin (best minus second best)
                    best_name = None
                    best_score = -1.0
                    second_best_score = -1.0
                    for name, ref_emb in self.registered_embeddings.items():
                        sim = float(np.dot(embedding, ref_emb) / (np.linalg.norm(embedding) * np.linalg.norm(ref_emb) + 1e-8))
                        if sim > best_score:
                            second_best_score = best_score
                            best_score = sim
                            best_name = name
                        elif sim > second_best_score:
                            second_best_score = sim
                    margin_ok = True
                    if len(self.registered_embeddings) > 1 and second_best_score >= 0:
                        margin_ok = (best_score - second_best_score) >= self.margin_threshold
                    
                    # Update bbox history for stability checks
                    if bbox:
                        x_min, y_min, y_max, x_max = bbox[0], bbox[1], bbox[3], bbox[2]
                        x_min, y_min, x_max, y_max = bbox
                        cx = (x_min + x_max) / 2.0
                        cy = (y_min + y_max) / 2.0
                        area = max(1.0, float((x_max - x_min) * (y_max - y_min)))
                        bbox_history.append((cx, cy, area))
                    else:
                        bbox_history.clear()
                    
                    # Update confirmation streak based on recognition, confidence, and margin
                    if recognized_name and confidence >= self.similarity_threshold and margin_ok:
                        if recognized_streak_name == recognized_name:
                            recognized_streak_count += 1
                            streak_similarities.append(confidence)
                        else:
                            recognized_streak_name = recognized_name
                            recognized_streak_count = 1
                            streak_similarities.clear()
                            streak_similarities.append(confidence)
                    else:
                        recognized_streak_name = None
                        recognized_streak_count = 0
                        streak_similarities.clear()
                        bbox_history.clear()
                    
                    # Mark attendance only after sufficient consecutive confirmations
                    if (recognized_streak_name and recognized_streak_count >= self.consecutive_match_required):
                        # Require robust confidence across the whole streak
                        if len(streak_similarities) >= self.consecutive_match_required:
                            streak_array = np.array(streak_similarities, dtype=np.float32)
                            median_streak_conf = float(np.median(streak_array))
                            frac_high = float(np.mean(streak_array >= self.mark_similarity_threshold))
                        else:
                            median_streak_conf = confidence
                            frac_high = 0.0
                        
                        # Bbox stability gating: very low movement and size variation across streak
                        stable_ok = False
                        if len(bbox_history) >= self.consecutive_match_required:
                            cxs = np.array([b[0] for b in bbox_history])
                            cys = np.array([b[1] for b in bbox_history])
                            areas = np.array([b[2] for b in bbox_history])
                            center_std = max(np.std(cxs), np.std(cys))
                            area_cv = (np.std(areas) / (np.mean(areas) + 1e-8))
                            # Tight gating: <= 6px jitter and <= 7% area change
                            stable_ok = (center_std <= self.bbox_center_std_threshold) and (area_cv <= self.bbox_area_cv_threshold)
                        
                        # Current-frame guard: ensure present frame also confirms the same identity strongly
                        current_frame_ok = (recognized_name == recognized_streak_name and confidence >= self.mark_similarity_threshold)
                        
                        if (median_streak_conf >= self.mark_similarity_threshold and
                            frac_high >= 0.85 and stable_ok and current_frame_ok and
                            (current_time - last_marked_time > recognition_delay or
                             last_marked_person != recognized_streak_name)):
                            marked = mark_attendance(recognized_streak_name)
                            last_marked_person = recognized_streak_name
                            last_marked_time = current_time
                            # Count confirmed recognition events
                            recognition_count += 1
            else:
                # No face detected; clear streak and bbox history
                recognized_streak_name = None
                recognized_streak_count = 0
                streak_similarities.clear()
                bbox_history.clear()
            
            # Draw bounding box
            frame = self.draw_bounding_box_with_label(
                frame, bbox, label=recognized_name, confidence=confidence
            )
            
            # Draw statistics
            cv2.putText(frame, f"FPS: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Recognitions: {recognition_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Debug metrics overlay
            if best_score is not None and second_best_score is not None:
                margin_val = best_score - second_best_score
            else:
                margin_val = 0.0
            if len(streak_similarities) > 0:
                streak_array_dbg = np.array(streak_similarities, dtype=np.float32)
                debug_median = float(np.median(streak_array_dbg))
                debug_frac_high = float(np.mean(streak_array_dbg >= self.mark_similarity_threshold))
            else:
                debug_median = 0.0
                debug_frac_high = 0.0
            if len(bbox_history) >= 3:
                cxs_dbg = np.array([b[0] for b in bbox_history])
                cys_dbg = np.array([b[1] for b in bbox_history])
                areas_dbg = np.array([b[2] for b in bbox_history])
                center_std_dbg = max(np.std(cxs_dbg), np.std(cys_dbg))
                area_cv_dbg = (np.std(areas_dbg) / (np.mean(areas_dbg) + 1e-8))
                debug_stable_ok = (center_std_dbg <= self.bbox_center_std_threshold) and (area_cv_dbg <= self.bbox_area_cv_threshold)
            else:
                center_std_dbg = 0.0
                area_cv_dbg = 0.0
                debug_stable_ok = False
            debug_current_ok = (recognized_name is not None and confidence >= self.mark_similarity_threshold)
            cv2.putText(frame, f"Best:{(best_score if best_score is not None else 0.0):.3f} 2nd:{(second_best_score if second_best_score is not None else 0.0):.3f} Margin:{margin_val:.3f} OK:{margin_ok}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
            cv2.putText(frame, f"Streak:{recognized_streak_count}/{self.consecutive_match_required} Median:{debug_median:.3f} High>={self.mark_similarity_threshold:.3f}:{debug_frac_high:.2f}", (10, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
            cv2.putText(frame, f"StableOK:{debug_stable_ok} CStd<={self.bbox_center_std_threshold:.1f} ACV<={self.bbox_area_cv_threshold:.2f} CurrOK:{debug_current_ok}", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
            
            # Draw registered faces list
            cv2.putText(frame, "Registered Faces:", (10, h - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            registered = get_registered_faces()
            for idx, name in enumerate(registered[:10]):  # Show first 10
                y_pos = h - 100 + (idx * 18)
                text = f"  - {name}"
                if recognized_name == name:
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if len(registered) > 10:
                cv2.putText(frame, f"  ... and {len(registered) - 10} more",
                           (10, h - 100 + 10 * 18), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (200, 200, 200), 1)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Allow quitting from the terminal (PowerShell) as well
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'q', b'Q', b'\x1b'):
                    print("\nClosing recognition system...")
                    break
            
            # Check for exit (q/Q/ESC) and window close
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\nClosing recognition system...")
                break
            
            # If user closed the window (clicking the X), exit gracefully
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed. Exiting...")
                break
            
            frame_count += 1
        
        # Cleanup resources
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nSession Summary:")
        print(f"  - Total frames processed: {frame_count}")
        print(f"  - Total recognitions: {recognition_count}")
        print("Thank you for using the Face Recognition Attendance System!\n")


def main():
    """
    Main function to run face recognition.
    """
    # Initialize with default threshold
    system = FaceRecognitionAttendance(similarity_threshold=0.7)
    
    # Start recognition
    system.start_recognition()


if __name__ == "__main__":
    main()
