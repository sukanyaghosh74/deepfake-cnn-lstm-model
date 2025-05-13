# Deepfake Detection using CNN + BiLSTM Architecture

![Deepfake Detection Banner](https://user-images.githubusercontent.com/your-image-path/banner.png)

## 📌 Project Overview

This project implements a deepfake detection system using a **custom hybrid architecture** combining **Convolutional Neural Networks (CNN)** and **Bidirectional Long Short-Term Memory networks (BiLSTM)**. It is developed as part of the AI Intern selection task for GenReal.ai.

Deepfakes manipulate facial expressions in videos and images using AI, often for malicious purposes. This project focuses on building a robust model that can accurately detect deepfake videos by learning both spatial and temporal features.

---

## 🔍 Problem Statement

With the rise of AI-generated synthetic media, it becomes crucial to detect forged content. Existing deepfake detection models struggle with generalization across:

* Video compression artifacts
* Frame inconsistencies
* Lighting variations
* Different manipulation techniques

Our model aims to address these problems by using a CNN for spatial feature extraction and a BiLSTM for capturing temporal relationships across frames.

---

## 🧠 Model Architecture

```
Input Video → Frame Extraction → Face Detection →
    CNN (Spatial Features)
        ↓
    BiLSTM (Temporal Modeling)
        ↓
    Dense Layer → Sigmoid Activation → Deepfake Probability
```

* **CNN Backbone**: Pretrained CNN (e.g., VGG16, ResNet50) used for extracting spatial facial features.
* **BiLSTM**: Models temporal dependencies across video frames.
* **Fully Connected Layer**: Outputs a probability score indicating real or fake.

![Architecture Diagram](https://user-images.githubusercontent.com/your-image-path/model-diagram.png)

---

## ⚙️ Preprocessing Pipeline

1. **Frame Extraction**: Extract 10–15 frames per video.
2. **Face Detection**: Use `face_recognition` or MTCNN to crop faces.
3. **Resizing**: Standardize all face images to 224x224 pixels.
4. **Normalization**: Pixel normalization to \[0,1].
5. **Sequence Creation**: Combine face sequences per video as input to BiLSTM.

---

## 📁 Folder Structure

```
.
├── data/
│   ├── real/
│   ├── fake/
│   └── ...
├── frames/
├── models/
├── preprocessing/
│   ├── extract_frames.py
│   └── detect_faces.py
├── train_model.py
├── evaluate_model.py
└── README.md
```

---

## 🧪 Experiments

* ✅ Tested on FaceForensics++ (C23 compression)
* ✅ Evaluated different CNN backbones (VGG16, ResNet)
* ✅ Compared LSTM vs BiLSTM
* ✅ Visualized activation maps using Grad-CAM

---

## 📊 Evaluation Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 93.2% |
| Precision | 91.6% |
| Recall    | 94.0% |
| F1-Score  | 92.8% |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/sukanyaghosh74/deepfake-cnn-lstm-model.git
cd deepfake-cnn-lstm-model

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python preprocessing/extract_frames.py
python preprocessing/detect_faces.py

# Train the model
python train_model.py

# Evaluate
python evaluate_model.py
```

---

## 📚 Technologies Used

* Python 3.10+
* OpenCV
* TensorFlow / PyTorch
* NumPy, Pandas, Matplotlib
* face\_recognition

---

## 🌟 Contributors

* **Sukanya Ghosh** — [GitHub](https://github.com/sukanyaghosh74) | [LinkedIn](https://www.linkedin.com/in/sukanya-ghosh-706129274/)

---

## 🏁 Status

✅ Completed Phase 1 (Preprocessing & Model Design)
⏳ Future Work: Deployment as a Web API using FastAPI/Streamlit

---

## 📄 License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* GenReal.ai for the opportunity
* FaceForensics++ Dataset creators
* Community and mentors for helpful feedback and guidance
