from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QSlider, QPushButton, QWidget
from motion_detector import MotionDetector
import numpy as np
import cv2
from skimage import filters, morphology, measure
import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide2.QtCore import Qt
from PySide2.QtGui import QImage, QPixmap

filename=""


class MainWindow(QMainWindow,):
    def __init__(self):
        # Call the parent constructor
        super(MainWindow, self).__init__()

        # Set the window title
        self.setWindowTitle("Object Tracker")

        # Create a central widget for the window
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create a vertical box layout for the central widget
        layout = QVBoxLayout()

        # Create a label to display the video frames
        self.image_label = QLabel()
        self.image_label.setMinimumSize(1024, 768)
        layout.addWidget(self.image_label)

        # Create a slider to navigate through the frames
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)  # Set maximum slider value according to the total number of frames in the video
        layout.addWidget(self.slider)

        # Create buttons to jump forward or backward in the video
        self.jump_forward_button = QPushButton("Jump Forward 60 Frames")
        layout.addWidget(self.jump_forward_button)

        self.jump_backward_button = QPushButton("Jump Backward 60 Frames")
        layout.addWidget(self.jump_backward_button)

        # Set the layout of the central widget
        self.central_widget.setLayout(layout)

        # Connect the UI components to their corresponding functions
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.jump_forward_button.clicked.connect(self.on_jump_forward_button_clicked)
        self.jump_backward_button.clicked.connect(self.on_jump_backward_button_clicked)

        # Open the video file and initialize the motion detector
        self.video_capture = cv2.VideoCapture(filename)
        self.motion_detector = MotionDetector(a=10, r=10, S=100, s=1, N=1)

        # Get the total number of frames in the video and set the maximum value of the slider accordingly
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.frame_count - 1)


    

    def on_slider_value_changed(self, value):
        # Set the position of the video capture to the current slider value
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, value)

        # Read the frame from the video capture
        ret, frame = self.video_capture.read()

        if ret:
            # Update the motion detector with the new frame
            self.motion_detector.update(frame)

            # Draw the bounding boxes for the tracked objects
            for tracked_object in self.motion_detector.tracked_objects:
                bbox = tracked_object["bbox"]
                cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)

            # Draw the trails for each tracked object
            for tracked_object in self.motion_detector.tracked_objects:
                history = tracked_object['history']
                for i in range(1, len(history)):
                    # Set the thickness of the trail based on its age
                    thickness = int(2 * (i / len(history)) + 1)
                    cv2.line(frame, history[i - 1], history[i], (0, 255, 0), thickness)

            # Convert the frame from BGR to RGB and display it in the image label
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)

            # Scale the pixmap to the size of the image label
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()

    def on_jump_forward_button_clicked(self):
        # Increment the slider value by 60 frames
        new_value = self.slider.value() + 60

        # Clamp the new value to the maximum slider value
        new_value = min(new_value, self.slider.maximum())

        # Set the new slider value
        self.slider.setValue(new_value)

    def on_jump_backward_button_clicked(self):
        # Decrement the slider value by 60 frames
        new_value = self.slider.value() - 60

        # Clamp the new value to the minimum slider value
        new_value = max(new_value, self.slider.minimum())

        # Set the new slider value
        self.slider.setValue(new_value)


if __name__ == "__main__":
    app = QApplication([])
    filename=sys.argv[1]
    window = MainWindow()
    window.show()

    app.exec_()
