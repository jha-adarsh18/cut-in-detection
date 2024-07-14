# Cut-In Detection System

This project uses YOLOv8 for real-time object detection to identify and warn about potential cut-in vehicles in a driving scenario. The system processes images from a specified directory, detects objects, calculates their distance and Time to Collision (TTC), and issues warnings if a cut-in is detected.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- statistics
- YOLOv8

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cut-in-detection.git
    cd cut-in-detection
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python-headless numpy ultralytics
    ```

3. Place your YOLOv8 model file (`best.pt`) in the desired directory.

## Usage

To run the script, use the following command:
    ```bash
    python script.py <image_directory>
    ```

Replace `<image_directory>` with the path to the directory containing your images.

## Example
    ```bash
    python script.py ./images
    ```

## Output

The output video will be saved as `output.avi` in the current directory. The video includes the detected objects and cut-in warnings.

### Playing the Video

You can play the generated video using any media player that supports AVI format. To play the video using VLC media player, use the following command:
    ```bash
    vlc output.avi
    ```

### Sample Video

Here is a sample of the cut-in detection video:

[output_detection.mp4](https://drive.google.com/file/d/1LqoqeMfKTTDJrTSdpOEBCHqktmPxnpd_/view?usp=sharing)


