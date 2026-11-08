import cv2
import os

# Define classes
classes = ["with_mask", "without_mask"]

# Create dataset folders
for cls in classes:
    os.makedirs(f"dataset/{cls}", exist_ok=True)

def open_camera():
    # On Windows, cv2.CAP_DSHOW prevents MSMF error -1072875772 (0xC00D3704)
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Connected to camera index {index} using DirectShow.")
                return cap
            cap.release()

    # Fallback to default backend
    for index in [0, 1]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Connected to camera index {index} using default backend.")
                return cap
            cap.release()

    return None

cap = open_camera()

if cap is None:
    print("\n[!] Could not connect to any webcam.")
    print("Troubleshooting checklist:")
    print(" 1. Check if another app (Zoom, Teams, Chrome, Camera App) is using the webcam.")
    print(" 2. Check Windows Settings -> Privacy & Security -> Camera -> Allow desktop apps.")
    print(" 3. If using an external USB webcam, try unplugging and reconnecting it.")
else:
    # Choose class to capture: "with_mask" or "without_mask"
    class_name = "with_mask"

    # Resume count based on existing images
    count = len(os.listdir(f"dataset/{class_name}"))
    print(f"\nCollecting images for class '{class_name}'.")
    print("Controls: Press 's' to save frame, 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame.")
            break

        # Show frame
        cv2.imshow("Capture - Press 's' to Save, 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            path = f"dataset/{class_name}/{class_name}_{count}.jpg"
            cv2.imwrite(path, frame)
            print(f"Saved: {path}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
