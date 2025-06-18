# Cheating Detection Using LSTM and Mediapipe

This project is a deep learning-based human action recognition system for detecting cheating behavior using LSTM networks and Mediapipe's holistic pose tracking. It performs real-time detection via webcam and is able to extract, train, evaluate, and predict suspicious activities like paper swapping or side glances during exams.

## Features

- **Action Detection with LSTM**
  - Recognizes temporal patterns in video using LSTM
  - Predicts predefined actions with confidence visualization
  - Real-time prediction from webcam input

- **Keypoint Extraction with Mediapipe**
  - Tracks face, pose, and hand landmarks using Mediapipe Holistic model
  - Converts landmarks into keypoint arrays (1662 total features per frame)
  - Stores keypoints as `.npy` files for training

- **Data Collection and Management**
  - Automated directory setup for collecting new action sequences
  - Dynamically organizes batches, actions, and frame sequences
  - Collects sequences per action and stores keypoints frame-by-frame

- **Model Evaluation and Training**
  - Builds and compiles an LSTM network using TensorFlow/Keras
  - Uses categorical cross-entropy and accuracy metrics
  - Provides test accuracy with softmax-based prediction scoring

- **Real-Time Inference and Visualization**
  - Displays predicted actions in a running sentence above the webcam feed
  - Live bar chart shows class probabilities for each prediction
  - Quits real-time feed with `q` key

- **Decision Tree Classifier**
  - Uses final predicted actions as input features
  - Determines whether the user is suspicious of cheating

## Technologies Used

- **Python** — Core implementation language
- **TensorFlow/Keras** — Deep learning framework for LSTM model
- **OpenCV** — Real-time webcam access and visualization
- **Mediapipe** — Landmark tracking for body, face, and hands
- **NumPy** — Keypoint data management and numerical ops
- **Scikit-learn** — Decision tree classifier and metrics
- **Pickle** — Model serialization

## Project Structure

- `mediapipe_detection()` — Preprocess and detect landmarks using Mediapipe
- `draw_landmarks()` — Visualize face/pose/hand landmarks
- `extract_keypoints()` — Convert Mediapipe results into flattened NumPy arrays
- `collect_keypoints_to_path()` — Record webcam sequences and save `.npy` data
- `load_data()` — Aggregate multiple data batches into training/testing datasets
- `build_and_compile_model()` — Constructs LSTM network for sequence classification
- `test_model()` — Prints classification accuracy on test set
- `test_real_time()` — Runs webcam prediction loop with visual feedback
- `extract_info()` — Summarizes behavior counts from predictions
- `save_lstm_model()` / `load_lstm()` — Saves and loads trained LSTM model
- `load_decision_tree()` — Loads pickled decision tree for cheating evaluation

## How to Use

1. **Collect Data from Webcam**
   - Run the data collection script
   - Actions will be saved into structured folders like `Actions Dataset/1/<ACTION>/<SEQ>/frame.npy`

2. **Train the LSTM Model**
   ```python
   X_train, y_train, X_test, y_test = load_data()
   model = build_and_compile_model()
   model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
   save_lstm_model(model, "lstm_model.h5")
   ```

3. **Test the LSTM Model**
   ```python
   model = load_lstm("lstm_model.h5")
   test_model(model)
   ```

4. **Real-Time Detection**
   ```python
   model = load_lstm("lstm_model.h5")
   test_real_time(model)
   ```

5. **Run Final Prediction using Decision Tree**
   ```python
   actions_detected = ['Side Glance', 'Paper Swap', ...]
   paper_swap, side_glance = extract_info(actions_detected)
   record = [[paper_swap, side_glance]]
   tree = load_decision_tree("decision_tree_model.pkl")
   print(pred(record, tree))
   ```

## Input Format

- **Webcam Video Feed**
  - Captures 30-frame sequences for each action sample
- **Keypoint Format**
  - Each `.npy` file stores 1662 floats (33 pose × 4 + 468 face × 3 + 21 left hand × 3 + 21 right hand × 3)

## Output Design

- **Saved Data**
  - Folder structure per batch: `batch_id/action_name/sequence_id/frame_id.npy`
- **Training Output**
  - Test accuracy printed after training
- **Prediction Output**
  - Real-time sentence display of actions on video
  - Colored probability bars for each action
- **Cheating Result**
  - Text result from decision tree: whether the behavior is suspicious

## Dependencies

- TensorFlow >= 2.0
- Mediapipe
- OpenCV
- NumPy
- Scikit-learn

Install all Python dependencies with:

```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn
```

## License

MIT License — feel free to use and extend.
