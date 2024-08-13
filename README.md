# Computer Vision Project for Football  Analysis

This project uses computer vision techniques to analyze a football match and extract relevant information such as player and ball tracking, team assignment, speed and distance estimation, ball acquisition and camera movement estimation.

![[input_videos\Untitled video - Made with Clipchamp.gif]]
![[input_videos\Untitled video - Made with Clipchamp (1).gif]]

## Features

* Player and ball tracking using fine tuned YOLOv5 on custom dataset  
* Team assignment using color segmentation (KNN)
* Speed and distance estimation using  **perspective transformation**.
* Camera movement estimation using **Lucas-Kanade Optical Flow**

## Requirements

* Python 3.8+
* OpenCV 4.5+
* NumPy 1.20+
* scikit-learn 0.24+
* Pandas 1.3+
* ultralytics 8.0+
