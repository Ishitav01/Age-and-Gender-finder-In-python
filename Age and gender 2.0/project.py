import cv2
import math

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'C:/Users/ISHU/Desktop/Python project/Age and gender 2.0/haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe("C:/Users/ISHU/Desktop/Python project/Age and gender 2.0/deploy_age.prototxt", "C:/Users/ISHU/Desktop/Python project/Age and gender 2.0/age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("C:/Users/ISHU/Desktop/Python project/Age and gender 2.0/deploy_gender.prototxt", "C:/Users/ISHU/Desktop/Python project/Age and gender 2.0/gender_net.caffemodel")

# Function to calculate age based on predicted age class
def calculate_age(age_predictions):
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
    max_index = age_predictions[0].argmax()
    age_label = age_list[max_index]
    return age_label

# Function to calculate gender based on predicted gender class
def calculate_gender(gender_predictions):
    gender_list = ['Male', 'Female']
    gender_label = gender_list[gender_predictions[0].argmax()]
    return gender_label

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess face ROI for age classification
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Feed face ROI to age classifier
        age_net.setInput(blob)
        age_predictions = age_net.forward()
        age_label = calculate_age(age_predictions)

        # Preprocess face ROI for gender classification
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Feed face ROI to gender classifier
        gender_net.setInput(blob)
        gender_predictions = gender_net.forward()
        gender_label = calculate_gender(gender_predictions)

        # Draw bounding box and labels on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{gender_label}, {age_label}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
