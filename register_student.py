import cv2
import os

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

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

def register_student():
    name = input("Enter student name (e.g., Ali, John, Sara): ").strip()
    if not name:
        print("Invalid name. Registration cancelled.")
        return

    # Count existing photos for this student
    existing = [f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(name)]
    idx = len(existing) + 1
    save_path = os.path.join(KNOWN_FACES_DIR, f"{name}_{idx}.jpg")

    cap = open_camera()
    if cap is None:
        print("[!] Error: Could not connect to webcam.")
        return

    print(f"\nRegistering: {name}")
    print("Press 's' to capture photo, or 'q' to cancel.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        cv2.putText(
            frame,
            f"Registering: {name} (Press 's' to Save)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Register Student", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite(save_path, frame)
            print(f"[OK] Photo saved to {save_path}")
            break
        elif key == ord('q'):
            print("Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_student()

