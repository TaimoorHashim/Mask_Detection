import cv2
import os
import csv
import numpy as np
from datetime import datetime

# ==========================================
# SMART ATTENDANCE SYSTEM
# Deep Learning Facial Recognition (YuNet + SFace)
# ==========================================

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
YUNET_MODEL = "face_detection_yunet.onnx"
SFACE_MODEL = "face_recognition_sface.onnx"
COSINE_THRESHOLD = 0.45  # Standard threshold for SFace cosine similarity

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

if not os.path.exists(YUNET_MODEL) or not os.path.exists(SFACE_MODEL):
    raise FileNotFoundError(
        f"Model files not found. Ensure '{YUNET_MODEL}' and '{SFACE_MODEL}' exist in the project directory."
    )

detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.6)
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

def extract_face_feature(img):
    """Detects face and extracts SFace 128-d embedding feature vector."""
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    retval, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    face_box = faces[0]
    aligned_face = recognizer.alignCrop(img, face_box)
    feature = recognizer.feature(aligned_face)
    return feature

def load_known_faces():
    known_features = []
    known_names = []
    print("Loading student images from 'known_faces'...")

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read: {filename}")
            continue

        feature = extract_face_feature(image)
        if feature is None:
            print(f"Skipping {filename}: No clear face detected.")
            continue

        student_name = os.path.splitext(filename)[0]
        student_name = student_name.rsplit("_", 1)[0]

        known_features.append(feature)
        known_names.append(student_name)
        print(f" - Loaded: {student_name} ({filename})")

    return known_features, known_names

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])

    # Check if student is already marked today
    with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("Name") == name and row.get("Date") == today:
                return

    # Append attendance
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([name, today, current_time])

    print(f"[ATTENDANCE MARKED] {name} at {current_time}")

def open_camera():
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    for index in [0, 1]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    return None

def main():
    known_features, known_names = load_known_faces()

    if not known_features:
        print("\n[!] Warning: No valid face images found in 'known_faces' folder.")
        print("Please add student photos to 'known_faces/' or run 'python register_student.py'.\n")
        return

    print(f"\n[OK] Loaded {len(known_features)} student profile(s).\n")

    camera = open_camera()
    if camera is None:
        print("[!] Error: Webcam could not be opened. Check connection and permissions.")
        return

    print("==============================================")
    print("     Smart Attendance System Running")
    print("==============================================")
    print("Press 'q' to exit.\n")

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Failed to capture frame.")
                break

            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            retval, faces = detector.detect(frame)

            if faces is not None:
                for face in faces:
                    x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + fw), min(h, y + fh)

                    aligned_face = recognizer.alignCrop(frame, face)
                    current_feature = recognizer.feature(aligned_face)

                    best_score = -1.0
                    best_name = "Unknown"

                    for k_name, k_feat in zip(known_names, known_features):
                        score = recognizer.match(k_feat, current_feature, cv2.FaceRecognizerSF_FR_COSINE)
                        if score > best_score:
                            best_score = score
                            if score >= COSINE_THRESHOLD:
                                best_name = k_name

                    if best_name != "Unknown":
                        color = (0, 220, 0)  # Green
                        label_text = f"{best_name} ({best_score*100:.1f}%)"
                        mark_attendance(best_name)
                    else:
                        color = (0, 0, 220)  # Red
                        label_text = "Unknown"

                    # Draw bounding box and label banner
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - 30)), (x1 + max(140, fw), y1), color, -1)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1 + 6, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

            cv2.imshow("Smart Attendance System - Press 'q' to Exit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Attendance system closed.")

if __name__ == "__main__":
    main()
