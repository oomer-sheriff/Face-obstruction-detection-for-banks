# Mask Detection and Recording Application

This project provides a real-time mask detection application using OpenCV and TensorFlow. The application detects faces in a video feed, predicts whether the faces are wearing masks, and records video when no mask is detected.

## Features

- Real-time face and mask detection using OpenCV and TensorFlow.
- Records video when no mask is detected.
- Displays mask detection status and probability on the video feed.

## Prerequisites

- Python 3.x
- OpenCV
- TensorFlow
- NumPy

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/mask-detection-app.git
   cd mask-detection-app
   ```

2. **Install Required Libraries**

   ```bash
   pip install opencv-python tensorflow numpy
   ```

3. **Prepare the Mask Detection Model**

   - You need a trained mask detection model saved as `model.h5`. If you don't have one, you'll need to train a model or use a pre-trained model for mask detection.
   - Place the `model.h5` file in the project directory.

4. **Download the Haar Cascade File**

   - The code uses OpenCV's pre-trained Haar cascade for face detection, which is included in the OpenCV library. You don’t need to download it manually.

## Usage

1. **Run the Application**

   ```bash
   python app.py
   ```

2. **Video Feed and Mask Detection**

   - The application will start a video feed from your webcam.
   - It will display a bounding box around detected faces and show whether the person is wearing a mask or not.
   - The label will also include the confidence percentage.

3. **Recording Video**

   - If a face without a mask is detected, the application will start recording video and save it as `output.avi`.
   - If a mask is detected again, the recording will stop.

4. **Exit the Application**

   - Press the `q` key to exit the application and close the video feed.

## Code Overview

- `detect_and_predict_mask(frame, face_detector, mask_detector)`: Detects faces in a frame and predicts if they are wearing masks using the mask detection model.
- Video recording starts when a face without a mask is detected and stops when a mask is detected.
- Uses OpenCV for face detection and video handling, TensorFlow for mask prediction.

## Notes

- Ensure the mask detection model (`model.h5`) is properly trained and saved in the project directory.
- The application uses a Haar cascade for face detection. For better performance, consider using a more advanced face detection model if needed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)

For further questions or contributions, feel free to open an issue or submit a pull request.

Happy Mask Detecting!