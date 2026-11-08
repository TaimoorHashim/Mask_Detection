import os
import joblib
import cv2
import numpy as np
import gradio as gr

MODEL_PATH = "mask_detector.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run 'python train_model.py' first.")

data = joblib.load(MODEL_PATH)
model = data["model"]
classes = data["classes"]
image_size = data.get("image_size", (100, 100))

def predict_mask(image):
    if image is None:
        return "No image provided.", {}

    # Convert RGB (from Gradio) to BGR for OpenCV consistency
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = image_bgr.shape[:2]
    # Center crop region
    box_w, box_h = int(w * 0.5), int(h * 0.7)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    roi = image_bgr[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, image_size)
    features = (roi_resized.flatten() / 255.0).reshape(1, -1)

    # Predict
    probs = model.predict_proba(features)[0]
    confidences = {
        "Wearing Mask": float(probs[classes.index("with_mask")]),
        "No Mask": float(probs[classes.index("without_mask")])
    }

    pred_idx = np.argmax(probs)
    verdict = "✅ MASK DETECTED" if classes[pred_idx] == "with_mask" else "❌ NO MASK DETECTED"
    return verdict, confidences

demo = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="numpy", label="Upload or Capture Photo"),
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Label(label="Confidence Distribution")
    ],
    title="😷 AI Face Mask Detection",
    description="Upload a photo or take a picture with your webcam to test the mask classifier.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)

