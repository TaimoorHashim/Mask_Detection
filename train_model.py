import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATASET_DIR = "dataset"
MODEL_PATH = "mask_detector.pkl"
IMAGE_SIZE = (100, 100)

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Resize image to standardized dimension
    img_resized = cv2.resize(img, IMAGE_SIZE)
    # Normalize pixel values
    features = img_resized.flatten() / 255.0
    return features

def train():
    classes = ["with_mask", "without_mask"]
    X = []
    y = []

    print("Loading dataset...")
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"Error: Folder '{cls_dir}' not found.")
            return

        files = os.listdir(cls_dir)
        print(f" - Found {len(files)} images for '{cls}'")

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            feat = extract_features(fpath)
            if feat is not None:
                X.append(feat)
                y.append(label)

    if len(X) == 0:
        print("Error: No images found in dataset folders.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\nTotal samples loaded: {len(X)}")

    # Split dataset if enough samples exist
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_test)}")
    print("Training Random Forest Classifier...")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n[OK] Training Accuracy: {train_acc * 100:.1f}%")
    print(f"[OK] Validation Accuracy: {test_acc * 100:.1f}%")

    # Save model and classes
    payload = {
        "model": clf,
        "classes": classes,
        "image_size": IMAGE_SIZE
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\n[OK] Model saved successfully as '{MODEL_PATH}'!")

if __name__ == "__main__":
    train()
