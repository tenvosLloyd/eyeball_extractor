import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def detect_eyes_and_nose_bridge(image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find face landmarks
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print(f"No faces detected in image: {image_path}")
        return

    for face_landmarks in results.multi_face_landmarks:
        ih, iw, _ = image.shape

        # Define landmarks for left eye, right eye, and bridge of the nose
        left_eye_indices = [33, 133, 160, 158, 153, 144, 145, 153]
        right_eye_indices = [362, 263, 387, 385, 380, 373, 374, 380]
        nose_bridge_indices = [6]  # Nose tip

        # Extract coordinates for eyes and nose bridge
        left_eye = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) 
                    for i in left_eye_indices]
        right_eye = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) 
                     for i in right_eye_indices]
        nose_bridge = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) 
                       for i in nose_bridge_indices]

        # Crop the left and right eyes separately
        for eye, label in zip([left_eye, right_eye], ['left', 'right']):
            x_min = min([p[0] for p in eye])
            x_max = max([p[0] for p in eye])
            y_min = min([p[1] for p in eye])
            y_max = max([p[1] for p in eye])

            # Add some padding around the eyes
            padding = 5
            x_min = max(0, x_min - padding)
            x_max = min(iw, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(ih, y_max + padding)

            eye_img = image[y_min:y_max, x_min:x_max]

            if eye_img.size == 0:  # Check if the cropped image is valid
                print(f"Error cropping {label} eye in image: {image_path}")
                continue

            output_path = os.path.join(output_folder, f"{label}_eye_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, eye_img)
            print(f"{label.capitalize()} eye image saved to: {output_path}")

        # Combine eyes and nose bridge into a single image
        combined_indices = left_eye_indices + right_eye_indices + nose_bridge_indices
        combined_keypoints = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) 
                              for i in combined_indices]

        x_min_combined = min([p[0] for p in combined_keypoints])
        x_max_combined = max([p[0] for p in combined_keypoints])
        y_min_combined = min([p[1] for p in combined_keypoints])
        y_max_combined = max([p[1] for p in combined_keypoints])

        # Add padding for the combined image
        padding_x_combined = 10
        padding_y_combined = 20
        x_min_combined = max(0, x_min_combined - padding_x_combined)
        x_max_combined = min(iw, x_max_combined + padding_x_combined)
        y_min_combined = max(0, y_min_combined - padding_y_combined)
        y_max_combined = min(ih, y_max_combined + padding_y_combined)

        combined_eye_nose_img = image[y_min_combined:y_max_combined, x_min_combined:x_max_combined]

        if combined_eye_nose_img.size == 0:  # Check if the cropped image is valid
            print(f"Error cropping combined eye and nose bridge in image: {image_path}")
            continue

        combined_output_path = os.path.join(output_folder, f"combined_eyes_nose_{os.path.basename(image_path)}")
        cv2.imwrite(combined_output_path, combined_eye_nose_img)
        print(f"Combined eye and nose bridge image saved to: {combined_output_path}")


def process_images_in_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Process only image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            detect_eyes_and_nose_bridge(file_path, output_folder)


if __name__ == "__main__":
    input_folder = "./faces"  # Replace with your folder containing images
    output_folder = "./output_folder"  # Replace with your desired output folder

    process_images_in_folder(input_folder, output_folder)
