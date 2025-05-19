import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("ethnicity_model_1.h5")  # Change if your model file has a different name

# Define the race labels (edit if yours differ)
race_labels = ['White', 'Black', 'Asian', 'Indian', 'Other']

# Load OpenCV's Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocessing function
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))         # Resize to model input
    face_img = face_img.astype("float32") / 255.0     # Normalize
    face_img = np.expand_dims(face_img, axis=0)       # Add batch dimension
    return face_img                                   # Shape: (1, 64, 64, 3)

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face = frame[y:y+h, x:x+w]
        input_face = preprocess_face(face)

        # Predict race
        prediction = model.predict(input_face)[0]
        race_index = np.argmax(prediction)
        race = race_labels[race_index]
        confidence = prediction[race_index]

        # Draw bounding box and label
        label = f"{race} ({confidence*100:.2f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Ethnicity Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
