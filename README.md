# Eyeball Extractor

**Eyeball Extractor** is a Python script designed to detect and extract individual eyes and a combined region of both eyes along with the bridge of the nose from a folder of images. The script leverages Google's Mediapipe library for precise facial landmark detection.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Example Output](#example-output)
- [License](#license)
- [Contact](#contact)

## Overview

Eyeball Extractor processes each image in a specified folder to:
- Detect and extract the left and right eyes separately.
- Extract a combined region of both eyes and the bridge of the nose.

This tool is useful for facial analysis, biometric applications, and other computer vision tasks requiring focused eye regions.

## Requirements

- Python 3.6 or higher
- Mediapipe
- OpenCV
- NumPy

## Installation

1. **Clone the repository** or download the script:

   ```bash
   git clone https://github.com/yourusername/eyeball_extractor.git
   cd eyeball_extractor
   ```

2. **Install required Python libraries**:

   Make sure you have `pip` installed. Then run:

   ```bash
   pip install mediapipe opencv-python numpy
   ```

   If you are using a virtual environment (recommended), ensure it is activated before running the install command.

## Usage

1. **Prepare your images**:

   Place all the images from which you want to extract eyes and nose bridges in a specific folder (e.g., `input_images`).

2. **Run the script**:

   Update the `input_folder` and `output_folder` variables in the script with the paths to your input and output directories.

   ```python
   input_folder = "path/to/your/input/folder"  # Replace with your folder containing images
   output_folder = "path/to/your/output/folder"  # Replace with your desired output folder
   ```

   Run the script:

   ```bash
   python eyeball_extractor.py
   ```

3. **View the results**:

   The extracted eye images and combined eye and nose bridge images will be saved in the specified output folder.

## How It Works

1. **Face Detection and Landmark Extraction**: Eyeball Extractor uses Mediapipe's face mesh model to detect facial landmarks in each image. This model provides a robust detection mechanism using deep learning techniques.

2. **Eye and Nose Bridge Region Extraction**:
   - **Single Eye Cropping**: The script identifies landmarks corresponding to the left and right eyes and crops these regions individually.
   - **Combined Region Cropping**: Additionally, the script identifies a larger region encompassing both eyes and the bridge of the nose, then crops and saves this combined region.

3. **Saving Images**: The cropped eye regions and combined images are saved as separate files in the output folder.

## Example Output

Below are examples of the output images:

- **Left Eye**:
  ![Left Eye Example](example_output/left_eye_example.jpg)

- **Right Eye**:
  ![Right Eye Example](example_output/right_eye_example.jpg)

- **Combined Eyes and Nose Bridge**:
  ![Combined Eye and Nose Example](example_output/combined_eyes_nose_example.jpg)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [your-email@example.com](mailto:your-email@example.com).

## Acknowledgements

- [Mediapipe](https://google.github.io/mediapipe/) by Google for providing the face mesh model.
- [OpenCV](https://opencv.org/) for image processing.
