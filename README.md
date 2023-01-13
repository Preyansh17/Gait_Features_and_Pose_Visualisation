# Gait_Features_and_Pose_Visualisation
This repo contains implementations of several gait features along with features related to pose estimation

It also contains the code for a Flask Web Application which allows a user to record their video while walking, send it for Gait Analysis and then download their analysed video back from a webpage.

The Gait Analysis gives information about possible problems in a person's knee, hip or ankle joints.

## Description of Files

* Pose Visualisation Colab Notebook contains the Joint Visualisation given an image along with calculating the step length of a subject.

* app.py consists of the Flask Application which runs the Flask frontend as well as the backend for handling the requests and running the script.

* script.py consists of the Gait Analysis code which takes in an input video, does analysis on it and then outputs the video along with the analysis.

* index.html consists of the frontend code for the Flask Application which allows a user to record their video using their webcam, send it for analysis and then download the video along with the Gait Analysis.

* success.html is just a placeholder file without any use as such other than redirection in case of failure of POST request

## How to run the code on your system

First ensure that app.py, script.py and templates are in the same directory. File structure should be as follows:

-> app.py
-> script.py
-> templates
  -> index.html
  -> success.html
  
* Install flask
* Run app.py. It will open the frontend which you can access using the url given on the terminal.
* Make appropriate changes related to the file paths of the video files as well as urls in app.py, script.py and index.html. These are indicated by comments in the files
