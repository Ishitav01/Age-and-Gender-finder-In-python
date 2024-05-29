import cv2
import dlib



# Load pre-trained models
face_detector = dlib.get_frontal_face_detector()
gender_classifier = cv2.dnn.readNetFromCaffe("C:/Users/ISHU/Desktop/Python project/Age and gender/deploy_gender.prototxt" , "C:/Users/ISHU/Desktop/Python project/Age and gender/gender_net.caffemodel")
age_classifier = cv2.dnn.readNetFromCaffe("C:/Users/ISHU/Desktop/Python project/Age and gender/deploy_age.prototxt" , "C:/Users/ISHU/Desktop/Python project/Age and gender/age_net.caffemodel")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Get screen width and height
screen_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a full-screen window
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Function to calculate age based on predicted age class
def calculate_age(age_class):
    if age_class == 0:
        return "0-2"
    elif age_class == 1:
        return "4-6"
    elif age_class == 2:
        return "8-12"
    elif age_class == 3:
        return "15-20"
    elif age_class == 4:
        return "25-32"
    elif age_class == 5:
        return "38-43"
    elif age_class == 6:
        return "48-53"
    elif age_class == 7:
        return "60+"
    else:
        return "Unknown"

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    # Process each detected face
    for face in faces:
        # Get face bounding box coordinates
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess face ROI for gender classification
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Feed face ROI to gender classifier
        gender_classifier.setInput(blob)
        gender_predictions = gender_classifier.forward()
        gender_class = gender_predictions[0].argmax()

        # Preprocess face ROI for age classification
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Feed face ROI to age classifier
        age_classifier.setInput(blob)
        age_predictions = age_classifier.forward()
        age_class = age_predictions[0].argmax()

        # Convert gender and age predictions to readable labels
        gender_label = "Male" if gender_class == 1 else "Female"
        age_label = calculate_age(age_class)

        # Display gender and age labels on the frame
        cv2.putText(frame, f"{gender_label}, {age_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
cv2.waitKey(1) 