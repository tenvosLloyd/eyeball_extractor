import cv2

# Load the cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the image
image = cv2.imread('./faces/iStock-1264267722.jpg')  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Loop through the detected faces
for (x, y, w, h) in faces:
    # Draw a rectangle around the face (optional)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Extract the region of interest (ROI) for the face
    face_roi = gray[y:y+h, x:x+w]
    
    # Detect eyes within the face ROI
    eyes = eye_cascade.detectMultiScale(face_roi)
    
    eye_images = []
    
    # Loop through the detected eyes
    for i, (ex, ey, ew, eh) in enumerate(eyes):
        # Extract the eye region
        eye_roi = image[y+ey:y+ey+eh, x+ex:x+ex+ew]
        eye_images.append(eye_roi)
        
        # Save the individual eye images
        cv2.imwrite(f'eye_{i+1}.jpg', eye_roi)
        
    # Optionally, combine the two eye images into one image
    # if len(eye_images) == 2:
    #     combined_eyes = cv2.hconcat(eye_images)
    #     cv2.imwrite('combined_eyes.jpg', combined_eyes)

# Display the image with rectangles (optional)
cv2.imshow('Detected Face and Eyes', image)
cv2.waitKey(0)

