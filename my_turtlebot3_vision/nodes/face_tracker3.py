#!/usr/bin/env python

import rospy
import cv2
import cv2.cv as cv
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np


class FaceTracker(object):

    def __init__(self, node_name):
        self.node_name = node_name

        rospy.init_node(self.node_name)
        rospy.loginfo("Starting node " + str(self.node_name))

        rospy.on_shutdown(self.cleanup)

        # A number of parameters to determine what gets displayed on the screen. These can be overridden the appropriate launch file
        self.show_text = rospy.get_param("~show_text", True)
        self.show_boxes = rospy.get_param("~show_boxes", True)
        self.flip_image = rospy.get_param("~flip_image", False)   # TBD: do we need it?
        self.feature_size = rospy.get_param("~feature_size", 1)
        self.show_add_drop = rospy.get_param("~show_add_drop", False)

        # Get the paths to the cascade XML files for the Haar detectors.
        # These are set in the launch file.
        cascade_1 = rospy.get_param("~cascade_1", "")
        cascade_2 = rospy.get_param("~cascade_2", "")
        cascade_3 = rospy.get_param("~cascade_3", "")

        # Initialize the Haar detectors using the cascade files
        self.cascade_1 = cv2.CascadeClassifier(cascade_1)
        self.cascade_2 = cv2.CascadeClassifier(cascade_2)
        self.cascade_3 = cv2.CascadeClassifier(cascade_3)

        # Set cascade parameters that tend to work well for faces.
        # Can be overridden in launch file
        self.haar_scaleFactor = rospy.get_param("~haar_scaleFactor", 1.3)
        self.haar_minNeighbors = rospy.get_param("~haar_minNeighbors", 3)
        self.haar_minSize = rospy.get_param("~haar_minSize", 30)
        self.haar_maxSize = rospy.get_param("~haar_maxSize", 150)

        # Store all parameters together for passing to the detector
        self.haar_params = dict(scaleFactor=self.haar_scaleFactor,
                                minNeighbors=self.haar_minNeighbors,
                                flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                minSize=(self.haar_minSize, self.haar_minSize),
                                maxSize=(self.haar_maxSize, self.haar_maxSize))

        # Good features parameters
        self.gf_maxCorners = rospy.get_param("~gf_maxCorners", 200)
        self.gf_qualityLevel = rospy.get_param("~gf_qualityLevel", 0.02)
        self.gf_minDistance = rospy.get_param("~gf_minDistance", 7)
        self.gf_blockSize = rospy.get_param("~gf_blockSize", 10)
        self.gf_useHarrisDetector = rospy.get_param("~gf_useHarrisDetector", True)
        self.gf_k = rospy.get_param("~gf_k", 0.04)

        # Store all Good features parameters together for passing to the detector
        self.gf_params = dict(maxCorners=self.gf_maxCorners,
                              qualityLevel=self.gf_qualityLevel,
                              minDistance=self.gf_minDistance,
                              blockSize=self.gf_blockSize,
                              useHarrisDetector=self.gf_useHarrisDetector,
                              k=self.gf_k)

        # LK parameters
        self.lk_winSize = rospy.get_param("~lk_winSize", (10, 10))
        self.lk_maxLevel = rospy.get_param("~lk_maxLevel", 2)
        self.lk_criteria = rospy.get_param("~lk_criteria", (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

        # Store all LK parameters together for passing to the detector
        self.lk_params = dict(winSize=self.lk_winSize,
                              maxLevel=self.lk_maxLevel,
                              criteria=self.lk_criteria)

        # Face tracking parameters
        self.use_depth_for_tracking = rospy.get_param("~use_depth_for_tracking", False)
        self.min_keypoints = rospy.get_param("~min_keypoints", 20)
        self.abs_min_keypoints = rospy.get_param("~abs_min_keypoints", 6)
        self.std_err_xy = rospy.get_param("~std_err_xy", 2.5)
        self.pct_err_z = rospy.get_param("~pct_err_z", 0.42)
        self.max_mse = rospy.get_param("~max_mse", 10000)
        self.add_keypoint_distance = rospy.get_param("~add_keypoint_distance", 10)
        self.add_keypoints_interval = rospy.get_param("~add_keypoints_interval", 1)
        self.drop_keypoints_interval = rospy.get_param("~drop_keypoints_interval", 1)
        self.expand_roi_init = rospy.get_param("~expand_roi", 1.02)
        self.expand_roi = self.expand_roi_init
        self.face_tracking = True
        self.frame_index = 0

        # Initialize a number of global variables
        self.frame = None
        self.frame_size = None
        self.frame_width = None
        self.frame_height = None
        self.depth_image = None
        self.marker_image = None
        self.display_image = None

        self.grey = None
        self.prev_grey = None

        self.selected_point = None
        self.selection = None
        self.drag_start = None

        self.keypoints = list()
        self.detect_box = None
        self.track_box = None
        self.mask = None

        self.keystroke = None
        self.night_mode = False
        self.keep_marker_history = False

        # Cycle per second(number of processing loops per second.)
        self.cps = 0
        self.cps_values = list()
        self.cps_n_values = 20

        # Initialize the Region of Interest and its publisher
        self.ROI = RegionOfInterest()
        self.roi_pub = rospy.Publisher("/roi", RegionOfInterest, queue_size=1)

        # Create the main display window
        self.cv_window_name = self.node_name
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)

        # TBD: do we need to resize the window?

        # Create the cv_bridge object
        self.bridge = CvBridge()

        # Set a callback on mouse clicks on the image window
        cv.SetMouseCallback(self.cv_window_name, self.on_mouse_click, None)

        # Subscribe to the image and depth topics and set the appropriate callbacks
        # The image topic names can be remapped in the appropriate launch file
        self.image_sub = rospy.Subscriber("input_rgb_image", Image, self.image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber("input_depth_image", Image, self.depth_callback, queue_size=1)

    def on_mouse_click(self, event, x, y, flags, param):
        # This function allows the user to selection a ROI using the mouse
        if self.frame is None:
            return

        if event == cv.CV_EVENT_LBUTTONDOWN and not self.drag_start:
            self.track_box = None
            self.detect_box = None
            self.selected_point = (x, y)
            self.drag_start = (x, y)

        if self.drag_start:
            xmin = max(0, min(x, self.drag_start[0]))
            ymin = max(0, min(y, self.drag_start[1]))
            xmax = min(self.frame_width, max(x, self.drag_start[0]))
            ymax = min(self.frame_height, max(y, self.drag_start[1]))
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)

        if event == cv.CV_EVENT_LBUTTONUP:
            self.drag_start = None
            self.detect_box = self.selection

    def image_callback(self, ros_image):
        # Time this loop to get cycles per second
        start = time.time()

        # Convert the ROS Image to OpenCV format using a cv_bridge helper function
        self.frame = self.convert_image(ros_image)

        # Some webcams invert the image
        if self.flip_image:
            self.frame = cv2.flip(self.frame, 0)

        # Store the frame width and height in a pair of global variables
        if self.frame_width is None:
            self.frame_size = (self.frame.shape[1], self.frame.shape[0])
            self.frame_width, self.frame_height = self.frame_size

        # Create the marker image we will use for display purposes
        if self.marker_image is None:
            self.marker_image = np.zeros_like(self.frame)

        # Reset the marker image if we're not displaying the history
        if not self.keep_marker_history:
            self.marker_image = np.zeros_like(self.marker_image)

        # Process the image to detect and track objects or features
        self.processed_image = self.process_image(self.frame)

        # Display the user-selection rectangle or point
        self.display_selection()

        # Night mode: only display the markers
        if self.night_mode:
            self.processed_image = np.zeros_like(self.processed_image)

        # Merge the processed image and the marker image
        self.display_image = cv2.bitwise_or(self.processed_image, self.marker_image)

        # If we have a track box, then display it. The track box can be either a regular cvRect (x,y,w,h) or a rotated Rect (center, size, angle).
        if self.show_boxes:
            if self.track_box is not None and self.is_rect_nonzero(self.track_box):
                if len(self.track_box) == 4:
                    x, y, w, h = self.track_box
                    size = (w, h)
                    center = (x + w / 2, y + h / 2)
                    angle = 0
                    self.track_box = (center, size, angle)
                else:
                    (center, size, angle) = self.track_box

                # For face tracking, an upright rectangle look best
                if self.face_tracking:
                    pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
                    pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
                    cv2.rectangle(self.display_image, pt1, pt2, cv.RGB(50, 255, 50), self.feature_size, 8, 0)
                else:
                    # Otherwise display a rotated rectangle
                    vertices = np.int0(cv.BoxPoints(self.track_box))
                    cv2.drawContours(self.display_image, [vertices], 0, cv.RGB(50, 255, 50), self.feature_size)

            # If we don't yet have a track box, display the detect box if present
            elif self.detect_box is not None and self.is_rect_nonzero(self.detect_box):
                (pt1_x, pt1_y, w, h) = self.detect_box
                if self.show_boxes:
                    cv2.rectangle(self.display_image, (pt1_x, pt1_y), (pt1_x + w, pt1_y + h), cv.RGB(50, 255, 50), self.feature_size, 8, 0)

        # Publish the ROI
        self.publish_roi()

        # Compute the time for this loop and estimate CPS as a running average
        end = time.time()
        duration = end - start
        fps = int(1.0 / duration)
        self.cps_values.append(fps)
        if len(self.cps_values) > self.cps_n_values:
            self.cps_values.pop(0)
        self.cps = int(sum(self.cps_values) / len(self.cps_values))

        # Display CPS and image resolution if asked to
        if self.show_text:
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5

            # Print cycles per second (CPS) and resolution (RES) at top of the image
            if self.frame_size[0] >= 640:
                vstart = 25
                voffset = int(50 + self.frame_size[1] / 120.)
            elif self.frame_size[0] == 320:
                vstart = 15
                voffset = int(35 + self.frame_size[1] / 120.)
            else:
                vstart = 10
                voffset = int(20 + self.frame_size[1] / 120.)

            cv2.putText(self.display_image, "CPS: " + str(self.cps), (10, vstart), font_face, font_scale, cv.RGB(255, 255, 0))
            cv2.putText(self.display_image, "RES: " + str(self.frame_size[0]) + "X" + str(self.frame_size[1]), (10, voffset), font_face, font_scale, cv.RGB(255, 255, 0))

        # Update the image display
        cv2.imshow(self.cv_window_name, self.display_image)

        # Process any keyboard commands
        self.keystroke = cv2.waitKey(5)
        if self.keystroke is not None and self.keystroke != -1:
            try:
                cc = chr(self.keystroke & 255).lower()
                if cc == 'n':
                    self.night_mode = not self.night_mode
                elif cc == 'b':
                    self.show_boxes = not self.show_boxes
                elif cc == 't':
                    self.show_text = not self.show_text
                elif cc == 'c':
                    # Clear the current keypoints
                    self.keypoints = []
                    self.track_box = None
                    self.detect_box = None
                elif cc == 'd':
                    self.show_add_drop = not self.show_add_drop
                elif cc == 'q':
                    # They user has press the q key, so exit
                    rospy.signal_shutdown("User hit q key to quit.")
            except:
                pass

    def depth_callback(self, ros_image):
        pass

    def convert_image(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            return np.array(cv_image, dtype=np.uint8)
        except CvBridgeError, e:
            print e

    def convert_depth_image(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")

            # Convert to a numpy array since this is what OpenCV uses
            return np.array(depth_image, dtype=np.float32)
        except CvBridgeError, e:
            print e

    def publish_roi(self):
        if self.drag_start is not None:
            return

        if self.track_box is not None:
            roi_box = self.track_box
        elif self.detect_box is not None:
            roi_box = self.detect_box
        else:
            return

        try:
            roi_box = self.cvBox2D_to_cvRect(roi_box)
        except:
            return

        # Watch out for negative offsets
        roi_box[0] = max(0, roi_box[0])
        roi_box[1] = max(0, roi_box[1])

        try:
            ROI = RegionOfInterest()
            ROI.x_offset = int(roi_box[0])
            ROI.y_offset = int(roi_box[1])
            ROI.width = int(roi_box[2])
            ROI.height = int(roi_box[3])
            self.roi_pub.publish(ROI)
        except:
            rospy.loginfo("Publishing ROI failed")

    def process_image(self, cv_image):
        try:
            # Create a greyscale version of the image
            self.grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Equalize the grey histogram to minimize lighting effects
            self.grey = cv2.equalizeHist(self.grey)

            # STEP 1: Detect the face if we haven't already
            if self.detect_box is None:
                self.keypoints = list()
                self.track_box = None
                self.detect_box = self.detect_face(self.grey)
            else:
                # STEP 2: If we aren't yet tracking keypoints, get them now
                if not self.track_box or not self.is_rect_nonzero(self.track_box):
                    self.track_box = self.detect_box
                    self.keypoints = self.get_keypoints(self.grey, self.track_box)

                # Store a copy of the current grey image used for LK tracking
                if self.prev_grey is None:
                    self.prev_grey = self.grey

                # STEP 3: If we have keypoints, track them using optical flow
                self.track_box = self.track_keypoints(self.grey, self.prev_grey)

                # STEP 4: Drop keypoints that are too far from the main cluster
                if self.frame_index % self.drop_keypoints_interval == 0 and len(self.keypoints) > 0:
                    ((cog_x, cog_y, cog_z), mse_xy, mse_z, score) = self.drop_keypoints(self.abs_min_keypoints, self.std_err_xy, self.max_mse)
                    if score == -1:
                        self.detect_box = None
                        self.track_box = None
                        return cv_image

                # STEP 5: Add keypoints if the number is getting too low
                if self.frame_index % self.add_keypoints_interval == 0 and len(self.keypoints) < self.min_keypoints:
                    self.expand_roi *= self.expand_roi_init
                    self.add_keypoints(self.track_box)
                else:
                    self.frame_index += 1
                    self.expand_roi = self.expand_roi_init

            # Store a copy of the current grey image used for LK tracking
            self.prev_grey = self.grey
        except AttributeError:
            pass

        return cv_image

    def detect_face(self, input_image):
        # First check one of the frontal templates
        if self.cascade_1:
            faces = self.cascade_1.detectMultiScale(input_image, **self.haar_params)

        # If that fails, check the profile template
        if len(faces) == 0 and self.cascade_3:
            faces = self.cascade_3.detectMultiScale(input_image, **self.haar_params)

        # If that also fails, check a the other frontal template
        if len(faces) == 0 and self.cascade_2:
            faces = self.cascade_2.detectMultiScale(input_image, **self.haar_params)

        # The faces variable holds a list of face boxes.
        # If one or more faces are detected, return the first one.first
        if len(faces) > 0:
            face_box = faces[0]
        else:
            # If no face were detected, print the "LOST FACE" message on the screen
            if self.show_text:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(self.marker_image, "LOST FACE!", (int(self.frame_size[0] * 0.65), int(self.frame_size[1] * 0.9)), font_face, font_scale, cv.RGB(255, 50, 50))
                face_box = None

        return face_box

    def get_keypoints(self, input_image, detect_box):
        # Initialize the mask with all black pixels
        self.mask = np.zeros_like(input_image)

        # Get the coordinates and dimensions of the detect_box
        try:
            x, y, w, h = detect_box
        except:
            return None

        # Set the selected rectangle within the mask to white
        self.mask[y:y + h, x:x + w] = 255

        # Compute the good feature keypoints within the selected region
        keypoints = list()
        kp = cv2.goodFeaturesToTrack(input_image, mask=self.mask, **self.gf_params)
        if kp is not None and len(kp) > 0:
            for x, y in np.float32(kp).reshape(-1, 2):
                keypoints.append((x, y))

        return keypoints

    def add_keypoints(self, track_box):
        # Look for any new keypoints around the current keypoints

        # Begin with a mask of all black pixels
        mask = np.zeros_like(self.grey)

        # Get the coordinates and dimensions of the current track box
        try:
            ((x, y), (w, h), a) = track_box
        except:
            try:
                x, y, w, h = track_box
                x += w / 2
                y += h / 2
                a = 0
            except:
                rospy.loginfo("Track box has shrunk to zero...")
                return

        x = int(x)
        y = int(y)

        # Expand the track box to look for new keypoints
        w_new = int(self.expand_roi * w)
        h_new = int(self.expand_roi * h)

        pt1 = (x - int(w_new / 2), y - int(h_new / 2))
        pt2 = (x + int(w_new / 2), y + int(h_new / 2))

        mask_box = ((x, y), (w_new, h_new), a)

        # Display the expanded ROI with a yellow rectangle
        if self.show_add_drop:
            cv2.rectangle(self.marker_image, pt1, pt2, cv.RGB(255, 255, 0))

        # Create a filled white ellipse within the track box to define the ROI
        cv2.ellipse(mask, mask_box, cv.CV_RGB(255, 255, 255), cv.CV_FILLED)

        if self.keypoints is not None:
            # Mask the current keypoints
            for x, y in [np.int32(p) for p in self.keypoints]:
                cv2.circle(mask, (x, y), 5, 0, -1)

        new_keypoints = cv2.goodFeaturesToTrack(self.grey, mask=mask, **self.gf_params)

        # Append new keypoints to the current list if they are not too far from the current cluster
        if new_keypoints is not None:
            for x, y in np.float32(new_keypoints).reshape(-1, 2):
                distance = self.distance_to_cluster((x, y), self.keypoints)
                if distance > self.add_keypoint_distance:
                    self.keypoints.append((x, y))

                    # Briefly display a blue disc where the new point is added
                    if self.show_add_drop:
                        cv2.circle(self.marker_image, (x, y), 3, (255, 255, 0, 0), cv.CV_FILLED, 2, 0)

            # Remove duplicate keypoints
            self.keypoints = list(set(self.keypoints))

    def distance_to_cluster(self, test_point, cluster):
        min_distance = 10000
        for point in cluster:
            if point == test_point:
                continue
            # Use L1 distance since it is faster than L2
            distance = abs(test_point[0] - point[0]) + abs(test_point[1] - point[1])
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def drop_keypoints(self, min_keypoints, outlier_threshold, mse_threshold):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        sse = 0
        keypoints_xy = self.keypoints
        keypoints_z = self.keypoints
        n_xy = len(self.keypoints)
        n_z = n_xy

        # If there are no keypoints left to track, start over
        if n_xy == 0:
            return ((0, 0, 0), 0, 0, -1)

        # Compute the COG (center of gravity) of the cluster
        for point in self.keypoints:
            sum_x += point[0]
            sum_y += point[1]

        mean_x = sum_x / n_xy
        mean_y = sum_y / n_xy
        mean_z = 0

        if self.use_depth_for_tracking and self.depth_image is not None:
            for point in self.keypoints:
                try:
                    z = self.depth_image[point[1], point[0]]
                except:
                    n_z -= 1
                    continue

                if not isnan(z):
                    sum_z += z
                else:
                    n_z -= 1
                    continue

            try:
                mean_z = sum_z / n_z
            except:
                mean_z = -1

        else:
            mean_z = -1

        # Compute the x-y MSE (mean squared error) of the cluster in the camera plane
        for point in self.keypoints:
            sse += (point[0] - mean_x) * (point[0] - mean_x) + (point[1] - mean_y) * (point[1] - mean_y)

        # Get the average over the number of feature points
        mse_xy = sse / n_xy

        # The MSE must be > 0 for any sensible feature cluster
        if mse_xy == 0 or mse_xy > mse_threshold:
            return ((0, 0, 0), 0, 0, -1)

        # Throw away the outliers based on the x-y variance
        max_err = 0
        for point in self.keypoints:
            std_err = ((point[0] - mean_x) * (point[0] - mean_x) + (point[1] - mean_y) * (point[1] - mean_y)) / mse_xy
            if std_err > max_err:   # mayuan: this can be removed
                max_err = std_err
            if std_err > outlier_threshold:
                keypoints_xy.remove(point)
                if self.show_add_drop:
                    # Briefly mark the removed points in red
                    cv2.circle(self.marker_image, (point[0], point[1]), 3, (0, 0, 255), cv.CV_FILLED, 2, 0)
                try:
                    keypoints_z.remove(point)
                    n_z -= 1
                except:
                    pass

                n_xy -= 1

        # Now do the same for depth
        if self.use_depth_for_tracking and self.depth_image is not None:
            sse = 0
            for point in keypoints_z:
                try:
                    z = self.depth_image[point[1], point[0]]
                    sse += (z - mean_z) * (z - mean_z)
                except:
                    n_z -= 1

            try:
                mse_z = sse / n_z
            except:
                mse_z = 0

            # Throw away the outliers based on depth using percent error rather than
            # standard error since depth values can jump dramatically at object
            # boundaries
            for point in keypoints_z:
                try:
                    z = self.depth_image[point[1], point[0]]
                except:
                    continue

                try:
                    pct_err = abs(z - mean_z) / mean_z
                    if pct_err > self.pct_err_z:
                        keypoints_xy.remove(point)
                        if self.show_add_drop:
                            # Briefly mark the removed points in red
                            cv2.circle(self.marker_image, (point[0], point[1]), 2, (0, 0, 255), cv.CV_FILLED)
                except:
                    pass
        else:
            mse_z = -1

        self.keypoints = keypoints_xy

        # Consider a cluster bad if we have fewer than min_keypoints left
        if len(self.keypoints) < min_keypoints:
            score = -1
        else:
            score = 1

        return ((mean_x, mean_y, mean_z), mse_xy, mse_z, score)

    def track_keypoints(self, grey, prev_grey):
        # We are tracking points between the previous frame and the current frame
        img0, img1 = prev_grey, grey

        # Reshape the current keypoints into a numpy array required by calcOpticalFlowPyrLK()
        p0 = np.float32([p for p in self.keypoints]).reshape(-1, 1, 2)

        # Calculate the optical flow from the previous frame to the current frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)

        # Do the reverse calculation: from the current frame to the previous frame
        try:
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)

            # Compute the distance between corresponding points in the two flows
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)

            # If the distance between pairs of points is < 1 pixel, set a value in the
            # "good" array to True, otherwise False
            good = d < 1

            # Initialize a list to hold new keypoints
            new_keypoints = list()

            # Cycle through all current and new keypoints and only keep those that satisfy the "good" condition above
            for (x, y), good_flag in zip(p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                new_keypoints.append((x, y))

                # Draw the keypoint on the image
                cv2.circle(self.marker_image, (x, y), self.feature_size, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)

            self.keypoints = new_keypoints

            # Convert the keypoints list to a numpy array
            keypoints_array = np.float32([p for p in self.keypoints]).reshape(-1, 1, 2)

            # If we have enough points, find the best fit ellipse around them
            if len(self.keypoints) > 6:
                track_box = cv2.fitEllipse(keypoints_array)
            else:
                # Otherwise, find the best fitting rectangle
                track_box = cv2.boundingRect(keypoints_array)
        except:
            track_box = None

        return track_box

    def process_depth_image(self, cv_image):
        return cv_image

    def display_selection(self):
        # If the user is selecting a region with the mouse, display the corresonding rectangle for feedback.
        if self.drag_start and self.is_rect_nonzero(self.selection):
            x, y, w, h = self.selection
            cv2.rectangle(self.marker_image, (x, y), (x + w, y + h), (0, 255, 255), self.feature_size)
            self.selected_point = None

        # Else if the user has clicked on a point on the image, display it as a small circle.
        elif not self.selected_point is None:
            x = self.selected_point[0]
            y = self.selected_point[1]
            cv2.circle(self.marker_image, (x, y), self.feature_size, (0, 255, 255), self.feature_size)

    def is_rect_nonzero(self, rect):
        # First assume a simple CRect type
        try:
            (_, _, w, h) = rect
            return (w > 0) and (h > 0)
        except:
            try:
                # Otherwise, assume a CvBox2D type
                ((_, _), (w, h), a) = rect
                return (w > 0) and (h > 0)
            except:
                return False

    def cvBox2D_to_cvRect(self, roi):
        try:
            if len(roi) == 3:
                (center, size, angle) = roi
                pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
                pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
                rect = [pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1]]
            else:
                rect = list(roi)
        except:
            return [0, 0, 0, 0]

        return rect

    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node_name = "face_tracker"
        FaceTracker(node_name)
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down face tracker node."
        cv.DestroyAllWindows()
