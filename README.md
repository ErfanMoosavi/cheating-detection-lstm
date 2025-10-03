# Cheating Detection Using LSTM and Mediapipe

## 📌 Overview

**Cheating Detection** is a **real-time human action recognition system** that detects suspicious behaviors during exams. It leverages **LSTM networks** for temporal sequence modeling and **Mediapipe** for holistic pose tracking. The system captures keypoints, trains models, and predicts cheating behavior automatically.

---

## ✨ Features

* **Action Detection with LSTM**

  * Recognizes temporal patterns in video using LSTM networks
  * Predicts predefined actions with confidence visualization
  * Real-time prediction from webcam input

* **Keypoint Extraction with Mediapipe**

  * Tracks face, pose, and hand landmarks using Mediapipe Holistic model
  * Converts landmarks into keypoint arrays (1662 total features per frame)
  * Stores keypoints as `.npy` files for training

* **Data Collection and Management**

  * Automated directory setup for collecting new action sequences
  * Dynamically organizes batches, actions, and frame sequences
  * Collects sequences per action and stores keypoints frame-by-frame

* **Model Evaluation and Training**

  * Builds and compiles an LSTM network using TensorFlow/Keras
  * Uses categorical cross-entropy and accuracy metrics
  * Provides test accuracy with softmax-based prediction scoring

* **Real-Time Inference and Visualization**

  * Displays predicted actions in a running sentence above the webcam feed
  * Live bar chart shows class probabilities for each prediction
  * Quits real-time feed with `q` key

* **Decision Tree Classifier**

  * Uses final predicted actions as input features
  * Determines whether the user is suspicious of cheating

---

## 🔍 Technologies Used

* **Python** — Core implementation language
* **TensorFlow/Keras** — Deep learning framework for LSTM model
* **OpenCV** — Real-time webcam access and visualization
* **Mediapipe** — Landmark tracking for body, face, and hands
* **NumPy** — Keypoint data management and numerical operations
* **Scikit-learn** — Decision tree classifier and metrics
* **Pickle** — Model serialization

---

## 🔹 Input Format

* **Webcam Video Feed**

  * Captures 30-frame sequences for each action sample
* **Keypoint Format**

  * Each `.npy` file stores 1662 floats (33 pose × 4 + 468 face × 3 + 21 left hand × 3 + 21 right hand × 3)

---

## ⚡ Dependencies

Install Python dependencies:

```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn
```

---

## 🤝 License

MIT License — feel free to use and extend.
