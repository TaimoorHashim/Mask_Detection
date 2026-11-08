import cv2
import numpy as np
import os
import time
import joblib

MODEL_PATH = "mask_detector.pkl"

def open_camera():
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Connected to camera index {index} via DirectShow.")
                return cap
            cap.release()

    for index in [0, 1]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Connected to camera index {index} via default backend.")
                return cap
            cap.release()
    return None

def run_detection():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file '{MODEL_PATH}' not found.")
        print("Please train your model first by running: python train_model.py")
        return

    print("Loading trained mask detection model...")
    data = joblib.load(MODEL_PATH)
    model = data["model"]
    classes = data["classes"]
    image_size = data.get("image_size", (100, 100))

    cap = open_camera()
    if cap is None:
        print("[!] Error: Could not open any webcam.")
        return

    print("\n==============================================")
    print("   AI Face Mask Detection System Active")
    print("==============================================")
    print("Controls:")
    print(" - Press 'q': Quit application")
    print(" - Press 'c': Capture and save screenshot\n")

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = int(1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        h, w = frame.shape[:2]

        # Target bounding box in frame center
        box_w, box_h = int(w * 0.45), int(h * 0.65)
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        # Extract ROI and prepare features
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, image_size)
        features = (roi_resized.flatten() / 255.0).reshape(1, -1)

        # Predict
        probs = model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        confidence = probs[pred_idx] * 100

        if pred_label == "with_mask":
            status_text = f"MASK ON ({confidence:.1f}%)"
            theme_color = (0, 200, 0)      # Green
        else:
            status_text = f"NO MASK ({confidence:.1f}%)"
            theme_color = (0, 0, 220)      # Red

        # Draw target box corners
        corner_len = 25
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), theme_color, 2)

        # Draw corner accents
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), theme_color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), theme_color, thickness)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), theme_color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), theme_color, thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), theme_color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), theme_color, thickness)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), theme_color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), theme_color, thickness)

        # Top status banner
        cv2.rectangle(frame, (x1, y1 - 38), (x2, y1), theme_color, -1)
        cv2.putText(
            frame,
            status_text,
            (x1 + 12, y1 - 12),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Top-left info HUD (FPS & instructions)
        cv2.rectangle(frame, (10, 10), (180, 50), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"FPS: {fps} | Press 'q' to quit",
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )

        cv2.imshow("Face Mask Detector - AI Batch 2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            screenshot_name = f"detection_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"[Captured] Screenshot saved: {screenshot_name}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()
