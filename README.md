StereoCam

#Functioning

Processes two images from two separate monocular cameras, calibrates images, and then uses those variables to recitfy and undistort inputted images 0_l.png and 0_r.png

#To Run

git clone this repository • create and activate a virutal environment that uses python3 • virtualenv -p python3 virtualenv_name • install required packages from requirements.txt using "pip install -r requirements.txt" • import the sample images left.png, right.png, left2.png, and right2.png to set initial distortion variables (or replace with your own images) • run python distort.py

test.py simply opens and displays video footage from each of the cameras

capture_frames.py opens video footage from each of the camera and saves it to left_pngs and right_pngs folders respectively
