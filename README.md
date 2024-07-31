# Pedestrian Tracking and Counting using YOLOv5 and OpenCV
This project demonstrates a pedestrian tracking and counting system implemented using the YOLOv5 object detection model, OpenCV for image processing, and a custom tracker to keep count of pedestrians crossing a designated area in a video feed.

# Project Overview
This project utilizes the YOLOv5 model to detect and track pedestrians in a CCTV video feed. It counts the number of unique individuals who cross a predefined area (Region of Interest, ROI) in the frame. The code leverages the capabilities of PyTorch, OpenCV, and a custom tracking algorithm to achieve this task.

# Installation
To run this project, you need to have Python 3.x installed along with the following libraries:

PyTorch

NumPy

OpenCV

YOLOv5

You can install the required Python packages using pip.

# Usage
Prepare the video file: Ensure you have a video file named cctv.mp4 in the same directory as the code. This video will be used for pedestrian detection and tracking.

Define the Region of Interest (ROI): Adjust the area array in the code to set the region of interest for tracking pedestrians. Modify these coordinates to fit your specific use case.

Run the script: Execute the Python script to start detecting, tracking, and counting pedestrians. The output will be displayed in a window showing the video feed with detected pedestrians and the count.

# Details of the Code
YOLOv5 Model: The YOLOv5 model is loaded using the torch.hub feature. It detects objects in the video feed and provides bounding box coordinates.

OpenCV for Video Processing: OpenCV is used to read the video feed, draw bounding boxes, and display the output.

Custom Tracker: The custom tracker is used to assign unique IDs to detected pedestrians and keep track of them across frames.

Region of Interest (ROI): The ROI is defined as a polygonal area in the frame where pedestrian counting is performed.
