import numpy as np
import cv2
from skimage import filters, morphology, measure
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide2.QtCore import Qt
from PySide2.QtGui import QImage, QPixmap
import math
from scipy.optimize import linear_sum_assignment

class MotionDetector:
    def __init__(self, a, r, S, s, N):
        self.a = a            # Sensitivity factor for background update
        self.r = r            # Threshold for motion detection
        self.S = S            # Minimum blob size for object detection
        self.s = s            # Scaling factor for motion detection
        self.N = N            # Maximum number of objects to track
        self.frames = []      # Buffer to store recent frames
        self.object_candidates = []   # Buffer to store candidate objects
        self.tracked_objects=[]       # List of currently tracked objects

    def preprocess_frame(self, frame):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to the grayscale frame
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        return blurred_frame

    def detect_motion(self, frame_t, frame_t_minus_1, frame_t_minus_2):
        # Compute the absolute difference between frame_t and frame_t_minus_1
        diff1 = cv2.absdiff(frame_t, frame_t_minus_1)
        
        # Compute the absolute difference between frame_t_minus_1 and frame_t_minus_2
        diff2 = cv2.absdiff(frame_t_minus_1, frame_t_minus_2)
        
        # Take the minimum of the two difference frames to get the motion frame
        motion_frame = cv2.min(diff1, diff2)
        
        # Threshold the motion frame to obtain binary motion mask
        _, motion_frame = cv2.threshold(motion_frame, self.r, 255, cv2.THRESH_BINARY)
        return motion_frame

    def dilate_motion_blobs(self, motion_frame):
        # Create a 9x9 kernel for dilation
        kernel = np.ones((9, 9), np.uint8)
        
        # Dilate the motion mask using the kernel
        dilated_frame = cv2.dilate(motion_frame, kernel, iterations=1)
        return dilated_frame

    def label_blobs(self, dilated_frame):
        # Label connected components in the dilated motion mask
        labeled_frame, num_labels = measure.label(dilated_frame, connectivity=2, return_num=True)
        return labeled_frame, num_labels

    def compute_bounding_boxes(self, labeled_frame, num_labels):
        # Create an empty list to store the coordinates of each object in the frame.
        object_candidates = []
        # Loop through each object label in the frame.
        for i in range(1, num_labels + 1):
            # Find the coordinates of all pixels with the current label.
            coords = np.argwhere(labeled_frame == i)
            # Find the minimum and maximum x and y values for the current label.
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Add the bounding box coordinates of the current label to the list.
            object_candidates.append(((x_min, y_min), (x_max, y_max)))
        # Return the list of bounding box coordinates for all objects in the frame.
        return object_candidates

    def update(self, frame):
        # Preprocess the current frame
        preprocessed_frame = self.preprocess_frame(frame)

        # Add the preprocessed frame to the list of frames
        self.frames.append(preprocessed_frame)

        # Remove the oldest frame if the list is longer than 3
        if len(self.frames) > 3:
            self.frames.pop(0)

        # If the list contains exactly 3 frames, continue with object detection
        if len(self.frames) == 3:
            # Get the current and two previous frames
            frame_t = self.frames[-1]
            frame_t_minus_1 = self.frames[-2]
            frame_t_minus_2 = self.frames[-3]

            # Detect motion in the current frame using the previous two frames
            motion_frame = self.detect_motion(frame_t, frame_t_minus_1, frame_t_minus_2)

            # Dilate the motion blobs to make them more visible
            dilated_frame = self.dilate_motion_blobs(motion_frame)

            # Label the motion blobs
            labeled_frame, num_labels = self.label_blobs(dilated_frame)

            # Compute the bounding boxes for each labeled object
            new_object_candidates = self.compute_bounding_boxes(labeled_frame, num_labels)

            # Create a new list of tracked objects based on the new object candidates
            new_tracked_objects = [{"centroid": ((bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2),
                            "bbox": bbox,
                            "history": []} for bbox in new_object_candidates]

            # Calculate the cost matrix based on the Euclidean distance between centroids
            cost_matrix = []
            for tracked_object in self.tracked_objects:
                row = []
                for new_tracked_object in new_tracked_objects:
                    distance = math.sqrt((tracked_object['centroid'][0] - new_tracked_object['centroid'][0]) ** 2 +
                                        (tracked_object['centroid'][1] - new_tracked_object['centroid'][1]) ** 2)
                    row.append(distance)
                cost_matrix.append(row)

            # Find the optimal object association using the Hungarian algorithm
            if cost_matrix:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Update the history of each tracked object based on the new associations
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r][c] < 50:  # Only update the history if the cost is below a threshold
                        new_tracked_objects[c]['history'] = self.tracked_objects[r]['history']

            # Update the history for each tracked object
            for new_tracked_object in new_tracked_objects:
                new_tracked_object['history'].append(new_tracked_object['centroid'])

                # Limit the length of the history to 30
                max_history_length = 30
                if len(new_tracked_object['history']) > max_history_length:
                    new_tracked_object['history'].pop(0)

            # Set the list of tracked objects to the new list of tracked objects
            self.tracked_objects = new_tracked_objects




class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise):
        # Initialize the filter parameters
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Initialize the state and covariance matrices
        self.state = np.array([[0], [0], [0], [0]], dtype=float)  # [x, y, vx, vy]
        self.covariance = np.eye(4) * 1e-3

        # Define the transition matrix and observation matrix
        self.transition_matrix = np.array([[1, 0, dt, 0],
                                           [0, 1, 0, dt],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], dtype=float)

        self.observation_matrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], dtype=float)

        # Define the process noise and measurement noise covariance matrices
        self.process_noise_cov = np.array([[dt**4/4, 0, dt**3/2, 0],
                                           [0, dt**4/4, 0, dt**3/2],
                                           [dt**3/2, 0, dt**2, 0],
                                           [0, dt**3/2, 0, dt**2]], dtype=float) * self.process_noise

        self.measurement_noise_cov = np.eye(2) * self.measurement_noise

    def predict(self):
        # Predict the next state based on the current state and the transition matrix
        self.state = self.transition_matrix @ self.state

        # Update the covariance matrix based on the process noise covariance matrix
        self.covariance = self.transition_matrix @ self.covariance @ self.transition_matrix.T + self.process_noise_cov

        # Return the predicted position (x, y)
        return self.state[:2]

    def update(self, measurement):
        # Calculate the innovation (difference between the measurement and the predicted observation)
        innovation = measurement - self.observation_matrix @ self.state

        # Calculate the innovation covariance
        innovation_covariance = self.observation_matrix @ self.covariance @ self.observation_matrix.T + self.measurement_noise_cov

        # Calculate the Kalman gain
        kalman_gain = self.covariance @ self.observation_matrix.T @ np.linalg.inv(innovation_covariance)

        # Update the state and covariance matrices based on the Kalman gain and the innovation
        self.state = self.state + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ self.observation_matrix @ self.covariance
