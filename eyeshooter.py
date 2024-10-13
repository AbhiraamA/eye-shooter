import cv2
import dlib
import numpy as np

# Initialize webcam
camera = cv2.VideoCapture(0)

# Initialize face detector and eye detector
dtctr = dlib.get_frontal_face_detector()
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Screen resolution (assuming 1920x1080 for this example)
screen_width, screen_height = 1920, 1080

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = dtctr(gray_frame)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Focus on the face region
        face_region = gray_frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

            # Extract the eye region
            eye_region = face_region[ey:ey + eh, ex:ex + ew]

            # Threshold to detect the pupil
            _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest contour, which should correspond to the pupil
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                cv2.circle(frame, (int(x + ex + cx), int(y + ey + cy)), int(radius), (0, 0, 255), 2)

                # Estimate gaze direction (map pupil position to screen coordinates)
                pupil_x_ratio = cx / ew
                pupil_y_ratio = cy / eh

                # Map to screen coordinates
                screen_x = int(pupil_x_ratio * screen_width)
                screen_y = int(pupil_y_ratio * screen_height)

                # Visualize the gaze point on the frame (scaled down to the frame's resolution)
                frame_height, frame_width = frame.shape[:2]
                display_x = int(screen_x * (frame_width / screen_width))
                display_y = int(screen_y * (frame_height / screen_height))
                cv2.circle(frame, (display_x, display_y), 10, (255, 255, 0), 2)

    cv2.imshow('Eye Shooter', frame)
    key = cv2.waitKey(1)

    if key == 27:  # Press ESC to exit
        break

camera.release()
cv2.destroyAllWindows()
