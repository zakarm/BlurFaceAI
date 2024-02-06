# BlurFaceAI

## Introduction
BlurFaceAI is a Python application that utilizes computer vision techniques to detect and blur faces in real-time video streams. It can be used to maintain privacy in video conferences or surveillance systems.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/zakarm/BlurFaceAI.git
    ```

2. Navigate to the project directory:
    ```bash
    cd BlurFaceAI
    ```

3. Install the required dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage
Run the `main.py` script to start the face blurring process. This script captures video from the default camera and applies Gaussian blur to detected faces in real-time.

```bash
python3 main.py
```

Press 'q' to exit the application.

## Requirements
- Python 3.x
- OpenCV
- dlib
- numpy
  
## Credits
Face detection is performed using OpenCV's Haar Cascade Classifier and dlib's facial landmark detector.
Gaussian blur is applied to detected faces for privacy preservation.
