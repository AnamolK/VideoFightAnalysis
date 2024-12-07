# UFC Video Analysis

UFC Video Analysis is a Python application that analyzes UFC fight videos using computer vision and machine learning. It provides real-time fight statistics, including strikes, takedowns, and annotations for enhanced analysis.

## Features

- **Object Detection**: Detects fighters in video frames using YOLOv8.
- **Pose Estimation**: Tracks body landmarks with MediaPipe Pose.
- **Strike Detection**: Identifies strikes based on arm and body movements.
- **Takedown Detection**: Detects takedowns using body orientation and hip movements.
- **Real-Time Visualization**: Displays bounding boxes, landmarks, and fight statistics on the video.

## Technologies Used

- Python (OpenCV, NumPy)
- YOLOv8 (Ultralytics)
- MediaPipe Pose

## Requirements

To run the project, you'll need:

- Python 3.8 or later
- Required libraries:
  - `opencv-python`
  - `mediapipe`
  - `ultralytics`
  - `numpy`

## Setup and Installation

1. Clone the repository:

   ```bash
    git clone https://github.com/AnamolK/VideoFightAnalysis.git
    cd VideoFightAnalysis
   ```

## Usage
Use any mp4 you desire for the fight data and then place it in the folder.
Run the program.


## Contributing

Feel free to fork this repository and submit pull requests. If you encounter any bugs or want to request new features, feel free to open an issue!

## Help Wanted
- The current model is somewhat glitchy for pose visualization so if someone knows how to implement it better possibly using training, that would be great ( I will try to update this as well)
- Consistent Fighter Tracking
- Current implemntaiton is rudimentary using only visual cues for strikes and takedowns and no ML, would love to collaborate on this project!
