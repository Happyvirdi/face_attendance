import cv2
import mediapipe as mp
import numpy as np
from utils import save_face_embedding, get_face_data_dir, load_all_face_embeddings, cosine_similarity
import os
from pathlib import Path


class FaceRegistration:
    def __init__(self):
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for higher accuracy
            min_detection_confidence=0.7
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        # Stricter duplicate thresholds to avoid false positives
        self.duplicate_similarity_threshold = 0.98
        self.frame_high_similarity_threshold = 0.90
        self.frame_high_similarity_ratio_required = 0.70
    
    def get_face_embedding(self, frame):
        """
        Extract face embedding from a frame using MediaPipe Face Mesh.
        Returns a normalized, downsampled shape descriptor to improve differentiation.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (468, 3)
            # Normalize by landmark bbox
            min_xy = np.min(pts[:, :2], axis=0)
            max_xy = np.max(pts[:, :2], axis=0)
            center = (min_xy + max_xy) / 2.0
            scale = np.maximum(max_xy - min_xy, 1e-6)
            pts_xy_norm = (pts[:, :2] - center) / scale
            # Normalize z by std
            z = pts[:, 2]
            z_std = np.std(z) if np.std(z) > 1e-6 else 1.0
            z_norm = (z - np.mean(z)) / z_std
            # Downsample
            idx = np.arange(0, pts.shape[0], 16)  # ~30 points
            xy_sel = pts_xy_norm[idx, :]
            z_sel = z_norm[idx]
            # Radial histogram
            radii = np.sqrt(np.sum(xy_sel**2, axis=1))
            hist, _ = np.histogram(radii, bins=8, range=(0.0, np.max(radii) + 1e-6))
            hist = hist.astype(np.float32)
            if np.sum(hist) > 0:
                hist /= np.sum(hist)
            emb = np.concatenate([xy_sel.flatten(), z_sel.flatten(), hist], dtype=np.float32)
            return emb
        
        return None
    
    def has_face_detected(self, frame):
        """
        Check if a face is detected in the frame.
        
        Args:
            frame: numpy array, video frame
        
        Returns:
            tuple: (has_face: bool, bounding_box: list or None)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if results.detections:
            # Get bounding box of first detected face
            detection = results.detections[0]
            h, w, c = frame.shape
            
            bbox = detection.location_data.relative_bounding_box
            x_min = max(0, int(bbox.xmin * w) - 10)
            y_min = max(0, int(bbox.ymin * h) - 10)
            x_max = min(w, int((bbox.xmin + bbox.width) * w) + 10)
            y_max = min(h, int((bbox.ymin + bbox.height) * h) + 10)
            
            return True, [x_min, y_min, x_max, y_max]
        
        return False, None
    
    def draw_bounding_box(self, frame, bbox, color=(0, 255, 0)):
        """
        Draw bounding box on frame.
        
        Args:
            frame: numpy array, video frame
            bbox: list, [x_min, y_min, x_max, y_max]
            color: tuple, BGR color
        
        Returns:
            numpy array: frame with bounding box
        """
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        return frame
    
    def compute_duplicate_stats(self, embeddings, registered_embeddings):
        """
        Compute per-frame similarity stats against registered embeddings to robustly
        detect duplicates (same person registered under different names).
        Returns: (best_name, best_stats_dict, all_stats_dict)
        best_stats_dict keys: 'median', 'avg', 'frac_high'
        """
        stats = {}
        for reg_name, reg_embed in registered_embeddings.items():
            sims = [cosine_similarity(e, reg_embed) for e in embeddings if e is not None]
            if not sims:
                continue
            median_sim = float(np.median(sims))
            avg_sim = float(np.mean(sims))
            frac_high = float(np.mean([s >= self.frame_high_similarity_threshold for s in sims]))
            stats[reg_name] = {
                'median': median_sim,
                'avg': avg_sim,
                'frac_high': frac_high,
            }
        if not stats:
            return None, {'median': 0.0, 'avg': 0.0, 'frac_high': 0.0}, {}
        best_name = max(stats.items(), key=lambda kv: kv[1]['median'])[0]
        return best_name, stats[best_name], stats

    def register_face(self, name, frames_to_capture=100, allow_override=False):
        """
        Register a new face.
        
        Args:
            name: string, person's name
            frames_to_capture: int, number of frames to capture and average
        
        Returns:
            bool: True if registration successful, False otherwise
        """
        # Validate name
        name = name.strip()
        if not name:
            print("Error: Name cannot be empty!")
            return False
        
        if ' ' in name:
            print("Warning: Using name with spaces. Consider using names without spaces.")
        
        # Check if already registered
        face_data_dir = get_face_data_dir()
        if os.path.exists(os.path.join(face_data_dir, f"{name}.npy")):
            print(f"Warning: {name} is already registered. This will overwrite the existing data.")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam!")
            return False
        
        print(f"\n{'='*60}")
        print(f"Face Registration for: {name}")
        print(f"{'='*60}")
        # Determine dynamic capture mode
        dynamic_mode = False
        dynamic_min_frames = 125
        dynamic_max_frames = 300
        check_every = 25
        if isinstance(frames_to_capture, str) and frames_to_capture.lower() == 'auto':
            dynamic_mode = True
        elif frames_to_capture is None:
            dynamic_mode = True
        elif isinstance(frames_to_capture, int) and frames_to_capture <= 0:
            dynamic_mode = True
        
        if dynamic_mode:
            print("Dynamic capture enabled.")
            print(f"Will capture between {dynamic_min_frames}-{dynamic_max_frames} frames and stop early once similarity is sufficiently differentiated.")
        else:
            print(f"Capturing {frames_to_capture} frames...")
        print("Instructions:")
        print("  - Position your face in the center of the frame")
        print("  - Move slightly to capture different angles")
        print("  - Keep good lighting")
        print("  - Press 'q' to quit early or wait for auto-capture")
        print(f"{'='*60}\n")
        
        embeddings = []
        # Load registered embeddings once for on-the-fly similarity feedback
        registered_embeddings = load_all_face_embeddings()
        frame_count = 0
        no_face_count = 0
        
        # Use dynamic capture window if enabled
        while (len(embeddings) < (dynamic_max_frames if dynamic_mode else frames_to_capture)):
            ret, frame = cap.read()
        
            if not ret:
                print("Error: Failed to read from webcam!")
                cap.release()
                return False
        
            frame = cv2.flip(frame, 1)  # Mirror the frame
        
            # Check for face and get bounding box
            has_face, bbox = self.has_face_detected(frame)
        
            if has_face:
                no_face_count = 0
        
                # Draw bounding box
                frame = self.draw_bounding_box(frame, bbox, color=(0, 255, 0))
        
                # Get embedding
                embedding = self.get_face_embedding(frame)
        
                if embedding is not None:
                    embeddings.append(embedding)
                    cv2.putText(frame, f"Captured: {len(embeddings)}/{(dynamic_max_frames if dynamic_mode else frames_to_capture)}",
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Live similarity feedback and dynamic early stop
                    if registered_embeddings and (len(embeddings) % check_every == 0):
                        best_name_live, best_stats_live, _ = self.compute_duplicate_stats(embeddings, registered_embeddings)
                        if best_name_live:
                            cv2.putText(frame, f"Best vs registered: {best_name_live} m={best_stats_live['median']:.3f} fh={best_stats_live['frac_high']:.2f}",
                                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        # If we've captured enough frames, check differentiation to stop early
                        if dynamic_mode and len(embeddings) >= dynamic_min_frames:
                            # Stop early once similarity drops below strong-duplicate levels
                            if (best_name_live is None) or (best_stats_live['median'] < 0.990) or (best_stats_live['frac_high'] < 0.95):
                                cv2.putText(frame, "Similarity differentiated; finishing capture...",
                                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                # Show the message briefly and then break out to finalize
                                cv2.imshow(f'Registering Face - {name}', frame)
                                cv2.waitKey(300)
                                break
                            else:
                                cv2.putText(frame, f"Still similar to {best_name_live}. Vary angle/lighting...",
                                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
                # Display stability feedback
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    w = max(1, x_max - x_min)
                    h = max(1, y_max - y_min)
                    cx, cy = x_min + w // 2, y_min + h // 2
                    cv2.putText(frame, f"Box: cx={cx}, cy={cy}, w={w}, h={h}",
                               (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
            else:
                no_face_count += 1
                cv2.putText(frame, "No face detected. Please position your face.",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if no_face_count > 30:  # Reset if no face for 30 frames
                    embeddings = []
                    no_face_count = 0
                    cv2.putText(frame, "Resetting... Please try again.",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
            # Display frame
            cv2.imshow(f'Registering Face - {name}', frame)
        
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Registration cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Average embeddings for stability
        averaged_embedding = np.mean(embeddings, axis=0)

        # Duplicate face check against existing embeddings using robust stats
        registered_embeddings = load_all_face_embeddings()
        best_name, best_stats, all_stats = self.compute_duplicate_stats(embeddings, registered_embeddings)
        # Provide debug info on top candidates
        if all_stats:
            sorted_candidates = sorted(all_stats.items(), key=lambda kv: kv[1]['median'], reverse=True)[:3]
            print("Top similarity candidates (median, frac_high):")
            for cand_name, cand_stats in sorted_candidates:
                print(f"  - {cand_name}: median={cand_stats['median']:.3f}, frac_high={cand_stats['frac_high']:.2f}")
        # Decide duplicate using stricter median and consistency criteria
        is_strong_duplicate = (
            best_name is not None and
            best_stats['median'] >= self.duplicate_similarity_threshold and
            best_stats['frac_high'] >= self.frame_high_similarity_ratio_required
        )
        is_moderate_similarity = (
            best_name is not None and
            (best_stats['median'] >= 0.92 or best_stats['frac_high'] >= 0.50)
        )
        if is_strong_duplicate and best_name != name:
            print(f"\nYour face appears already registered as '{best_name}'.")
            print(
                f"Detected median similarity={best_stats['median']:.2f}, high-similarity fraction={best_stats['frac_high']:.2f}."
            )
            if allow_override:
                print("Proceeding with registration under the new name (override enabled).")
            else:
                # Robust input handling: keep prompting until a valid yes/no is entered
                while True:
                    choice = input("Do you still want to register under the new name? (yes/no): ").strip().lower()
                    if choice in ('yes', 'y'):
                        print("Proceeding with registration under the new name by user choice.")
                        break
                    elif choice in ('no', 'n', ''):
                        print("Registration aborted to prevent using the same face for multiple candidates.")
                        return False
                    else:
                        print("Please type 'yes' or 'no' and press Enter.")
        elif is_strong_duplicate and best_name == name:
            print(
                f"Note: New capture strongly matches existing '{name}' record "
                f"(median={best_stats['median']:.2f}, frac_high={best_stats['frac_high']:.2f}). Updating embedding."
            )
        elif is_moderate_similarity and best_name != name:
            # Soft warning only; allow registration to proceed
            print(
                f"Warning: This face has moderate similarity to '{best_name}' "
                f"(median={best_stats['median']:.2f}, frac_high={best_stats['frac_high']:.2f}). Proceeding."
            )

        # Save embedding
        save_face_embedding(name, averaged_embedding)
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! Face registered for {name}")
        print(f"Captured and averaged {len(embeddings)} frames")
        print(f"{'='*60}\n")
        
        return True


def main():
    """
    Main function to run face registration.
    """
    print("\n" + "="*60)
    print("FACE REGISTRATION SYSTEM")
    print("="*60 + "\n")
    
    registrar = FaceRegistration()
    
    while True:
        name = input("Enter person's name (or 'quit' to exit): ").strip()
        
        if name.lower() == 'quit':
            print("Exiting registration system.")
            break
        
        if not name:
            print("Please enter a valid name.\n")
            continue
        
        # Allow override by typing a trailing '!' after the name
        allow_override = False
        if name.endswith('!'):
            allow_override = True
            name = name.rstrip('!').strip()
            print("Override enabled: will allow registration even if a strong duplicate is detected.")
        
        success = registrar.register_face(name, frames_to_capture='auto', allow_override=allow_override)
        
        if success:
            another = input("Register another person? (yes/no): ").strip().lower()
            if another != 'yes' and another != 'y':
                print("Thank you for using the registration system!")
                break
        else:
            print("Registration failed. Please try again.\n")


if __name__ == "__main__":
    main()
