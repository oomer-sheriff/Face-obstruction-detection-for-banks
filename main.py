import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the mask detection model (you'll need to train this or use a pre-trained model)
mask_detector = load_model('model.h5')

def detect_and_predict_mask(frame, face_detector, mask_detector):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    faces_list = []
    preds = []
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces_list.append(face)
        
    if len(faces_list) > 0:
        faces_list = np.array(faces_list, dtype="float32")
        preds = mask_detector.predict(faces_list, batch_size=32)
    
    return (faces, preds)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces and predict mask
    (faces, preds) = detect_and_predict_mask(frame, face_detector, mask_detector)
    
    for (face_box, pred) in zip(faces, preds):
        (x, y, w, h) = face_box
        (mask, withoutMask) = pred
        
        # Determine if a mask is detected
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Include probability in label
        label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        
        # Display label and bounding box on frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Start or stop recording based on mask detection
        if label.startswith("No Mask") and not recording:
            recording = True
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        elif label.startswith("Mask") and recording:
            recording = False
            if out is not None:
                out.release()
                out = None
    
    # Record frame if recording is active
    if recording and out is not None:
        out.write(frame)
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()