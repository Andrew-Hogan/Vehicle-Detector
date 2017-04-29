# Vehicle-Detector
This program takes an input video or image and outputs vehicle detections using a linear SVM and sliding windows.

First performs a Histogram of Oriented Gradients (HOG) feature extraction (along with color features and a histogram of colors) on a labeled training set of images and trains Linear SVM classifier.
Then implements a sliding windows technique to search a test image or video.
Then uses heatmaps over a series of frames to create hard vehicle detections and renders bounding boxes.

A full project writeup is available along with the render video.
