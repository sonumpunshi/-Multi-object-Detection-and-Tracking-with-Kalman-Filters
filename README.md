# Multi-object-Detection-and-Tracking-with-Kalman-Filters

Motion Detection: Developed a motion detection algorithm based on the differences between successive video frames. Used frame differences between two sets of consecutive frames to filter out noise and achieve more robust motion detection. Applied a threshold to further filter out noise and segment the detected motion pixels.

Image Processing: Enhanced motion blobs using image dilation with a 3x3 window, improving connectivity of the detected objects. Used the connected components algorithm to assign a class label to each pixel belonging to the same motion blob.

Object Tracking: Created a MotionDetector class that maintains a list of object candidates detected over time and updates the tracking with each new video frame. Implemented several hyperparameters, including frame hysteresis, motion threshold, distance threshold, and the maximum number of objects to track.

Kalman Filters: Implemented a KalmanFilter class to track individual objects, with each object represented by a separate Kalman filter. Used the motion detector's list of object candidates to update the Kalman filters. Developed a tracking system that adds new objects to the tracking list, updates the state of currently tracked objects, and removes inactive objects based on logic involving object activity, distances between proposals and predictions, and measurement update frequency.

GUI Integration: Created a graphical user interface (GUI) program using PySide2 that loads a video file from the command line. Integrated the motion detection and object tracking components into the GUI, allowing users to interactively track objects in the video sequence. Added navigation features to the GUI, including the ability to jump forward and backward by multiple frames. Visualized tracked objects with trails of their previous positions, providing a visual history of object movement.
