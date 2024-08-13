# Computer Vision Project for Football  Analysis

This project uses computer vision techniques to analyze a football match and extract relevant information such as player and ball tracking, team assignment, speed and distance estimation, ball acquisition and camera movement estimation.

<img src="https://github.com/a-mankush/football_analysis_cv/blob/main/sample_gifs/Untitled%20video%20-%20Made%20with%20Clipchamp.gif?raw=true">

<img src="https://github.com/a-mankush/football_analysis_cv/blob/main/sample_gifs/Untitled%20video%20-%20Made%20with%20Clipchamp%20(1).gif?raw=true">

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
