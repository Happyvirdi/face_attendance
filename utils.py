import numpy as np
import os
import csv
from datetime import datetime
from pathlib import Path


def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: numpy array of face embedding
        embedding2: numpy array of face embedding
    
    Returns:
        float: similarity score between -1 and 1 (higher is more similar)
    """
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    embedding1 = embedding1 / norm1
    embedding2 = embedding2 / norm2
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    return similarity


def euclidean_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: numpy array of face embedding
        embedding2: numpy array of face embedding
    
    Returns:
        float: distance score (lower is more similar)
    """
    return np.linalg.norm(embedding1 - embedding2)


def get_face_data_dir():
    """
    Get or create the face data directory.
    
    Returns:
        str: path to face_data directory
    """
    face_data_dir = "face_data"
    Path(face_data_dir).mkdir(exist_ok=True)
    return face_data_dir


def _convert_raw_landmarks_to_descriptor(raw_embedding):
    """
    Convert a raw (468*3) landmark embedding to the normalized, downsampled
    shape descriptor used by recognition/registration.
    """
    try:
        pts = raw_embedding.reshape(-1, 3).astype(np.float32)  # (468, 3)
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
    except Exception:
        return raw_embedding.astype(np.float32)


def save_face_embedding(name, embedding):
    """
    Save face embedding for a person.
    
    Args:
        name: string, person's name
        embedding: numpy array, face embedding
    """
    face_data_dir = get_face_data_dir()
    
    # Create a numpy file with person's name
    file_path = os.path.join(face_data_dir, f"{name}.npy")
    np.save(file_path, embedding)
    print(f"Face embedding saved for {name}")


def load_all_face_embeddings():
    """
    Load all registered face embeddings.
    Converts legacy raw landmark embeddings to the new descriptor for consistency.
    """
    face_data_dir = get_face_data_dir()
    embeddings_dict = {}
    
    if not os.path.exists(face_data_dir):
        return embeddings_dict
    
    for file_name in os.listdir(face_data_dir):
        if file_name.endswith(".npy"):
            name = file_name[:-4]
            file_path = os.path.join(face_data_dir, file_name)
            embedding = np.load(file_path)
            # Convert legacy raw landmark vectors (length ~1404)
            if embedding.ndim == 1 and embedding.size in (1404, 468*3):
                embedding = _convert_raw_landmarks_to_descriptor(embedding)
            else:
                embedding = embedding.astype(np.float32)
            embeddings_dict[name] = embedding
    
    return embeddings_dict


def recognize_face(test_embedding, registered_embeddings, threshold=0.6):
    """
    Recognize a face by comparing with registered embeddings.
    
    Args:
        test_embedding: numpy array, embedding of detected face
        registered_embeddings: dict, {name: embedding_array}
        threshold: float, similarity threshold (0.6-0.7 recommended)
    
    Returns:
        tuple: (recognized_name, confidence_score) or (None, 0)
    """
    if not registered_embeddings:
        return None, 0
    
    best_match = None
    best_score = -1
    
    for name, embedding in registered_embeddings.items():
        # Using cosine similarity (1.0 = identical, -1.0 = opposite)
        similarity = cosine_similarity(test_embedding, embedding)
        
        if similarity > best_score:
            best_score = similarity
            best_match = name
    
    # Apply threshold
    if best_score >= threshold:
        return best_match, best_score
    
    return None, best_score


def is_person_marked_today(name, attendance_file="attendance.csv"):
    """
    Check if a person has already been marked present today.
    
    Args:
        name: string, person's name
        attendance_file: string, path to attendance CSV file
    
    Returns:
        bool: True if person already marked today, False otherwise
    """
    if not os.path.exists(attendance_file):
        return False
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        with open(attendance_file, 'r') as f:
            reader = csv.DictReader(f)
            if reader is None:
                return False
            
            for row in reader:
                if row.get('Name', '').strip() == name and row.get('Date', '') == today:
                    return True
    except Exception as e:
        print(f"Error checking attendance: {e}")
    
    return False


def mark_attendance(name, attendance_file="attendance.csv"):
    """
    Mark attendance for a person in CSV file.
    
    Args:
        name: string, person's name
        attendance_file: string, path to attendance CSV file
    
    Returns:
        bool: True if marked successfully, False if already marked
    """
    # Check if already marked today
    if is_person_marked_today(name, attendance_file):
        print(f"{name} already marked present today!")
        return False
    
    # Prepare data
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    # Create file if it doesn't exist
    file_exists = os.path.exists(attendance_file)
    
    try:
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['Name', 'Date', 'Time'])
            
            # Write attendance record
            writer.writerow([name, date, time])
        
        print(f"Attendance marked for {name} at {time}")
        return True
    
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False


def get_registered_faces():
    """
    Get list of all registered face names.
    
    Returns:
        list: list of registered person names
    """
    face_data_dir = get_face_data_dir()
    names = []
    
    if not os.path.exists(face_data_dir):
        return names
    
    for file_name in os.listdir(face_data_dir):
        if file_name.endswith(".npy"):
            names.append(file_name[:-4])
    
    return sorted(names)
