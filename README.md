# 😷 Real-Time AI Face Mask Detection System

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.3-FF7C00?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat&logo=windows&logoColor=white)](https://microsoft.com/windows)

An end-to-end Computer Vision and Machine Learning solution designed to monitor and classify whether individuals are wearing face masks in real-time. Built with native OpenCV video streaming, high-speed Scikit-Learn classification, and an interactive Gradio web application.

---

## 🌟 Key Features

- **DirectShow Camera Integration (`cv2.CAP_DSHOW`)**: Resolves common Windows Media Foundation (`MSMF`) lockups and error `0xC00D3704` for responsive streaming.
- **Ultra-Fast Model Training**: Uses a Random Forest classifier that trains in under 2 seconds without requiring heavy GPU frameworks or unstable native DLLs.
- **Live HUD & Bounding Boxes**:
  - 🟩 **Green Banner**: Mask Detected with real-time confidence percentage.
  - 🟥 **Red Banner**: No Mask Detected with warning indicator.
  - 📊 Live FPS counter and target alignment guide.
- **Dual Form Factor**:
  - **Desktop Window**: Low-latency standalone desktop OpenCV application.
  - **Web Dashboard**: Interactive Gradio web interface supporting both drag-and-drop file uploads and browser webcam inputs.
- **Jupyter & Script Support**: Includes both standalone `.py` scripts and interactive `.ipynb` notebooks for experimentation.

---

## 📁 Repository Structure

```text
Mask Detection/
├── dataset/
│   ├── with_mask/              # Training images wearing a mask
│   └── without_mask/           # Training images without a mask
├── save_image.py               # Webcam capture script for dataset collection
├── save_image.ipynb            # Jupyter notebook for image collection
├── train_model.py              # Model training script (Random Forest)
├── train_model.ipynb           # Jupyter notebook for training
├── detect_mask_video.py        # Real-time desktop webcam detector
├── app_gradio.py               # Interactive browser-based web dashboard
├── mask_detector.pkl           # Serialized trained model & metadata
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python installed. The required packages can be installed via `pip`:

```powershell
pip install opencv-python scikit-learn numpy joblib gradio
```

---

## 📖 Usage Guide

### Step 1: Collect Dataset Images

To build or expand your dataset directly from your webcam:

```powershell
python save_image.py
```

- **<kbd>s</kbd>**: Save the current frame to the selected dataset class folder.
- **<kbd>q</kbd>**: Exit the capture window.
- *Tip*: Switch the target class inside `save_image.py` by changing `class_name = "with_mask"` or `class_name = "without_mask"`. The script automatically checks existing files to avoid overwriting.

---

### Step 2: Train the Classification Model

Train the model on your collected images:

```powershell
python train_model.py
```

**Output:**
```text
Loading dataset...
 - Found 11 images for 'with_mask'
 - Found 11 images for 'without_mask'

Total samples loaded: 22
Training samples: 17, Validation samples: 5
Training Random Forest Classifier...

[OK] Training Accuracy: 100.0%
[OK] Validation Accuracy: 100.0%

[OK] Model saved successfully as 'mask_detector.pkl'!
```

---

### Step 3: Run Real-Time Video Detection

Launch the live webcam detector:

```powershell
python detect_mask_video.py
```

#### Controls & Shortcuts:
| Key | Action |
| :---: | :--- |
| **<kbd>c</kbd>** | Capture and save a timestamped screenshot (`detection_screenshot_<timestamp>.jpg`) |
| **<kbd>q</kbd>** | Quit the application and release camera resources cleanly |

---

### Step 4: Launch Web Dashboard (Gradio)

For a web-based testing interface:

```powershell
python app_gradio.py
```

Your default web browser will open at **`http://127.0.0.1:7860`**:
- Upload any image file or take a snapshot with your browser's camera.
- View real-time predictions and probability distribution bar charts.

---

## 🛠️ Troubleshooting

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `can't grab frame. Error: -1072875772` | Windows Media Foundation (`MSMF`) conflict or locked device. | Solved: The scripts use `cv2.CAP_DSHOW` to bypass MSMF. Also close any background instances of Zoom, Teams, or the Windows Camera app. |
| `ImportError: DLL load failed` with TensorFlow | Python 3.14 compatibility mismatch with experimental TensorFlow C++ wheels. | Solved: Switched to Scikit-Learn Random Forest, which is fully compatible and requires no VC++ redistributable setup. |
| Webcam fails to open | Windows camera privacy toggle blocking terminal/apps. | Go to **Windows Settings > Privacy & Security > Camera** and enable **"Let desktop apps access your camera"**. |

---

## 📄 License

This project is created for educational and demonstration purposes. Feel free to modify and expand it for attendance monitoring, smart access gates, or public safety systems!
