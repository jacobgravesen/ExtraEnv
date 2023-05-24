from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import pyrealsense2 as rs
import numpy as np
import threading
import time
from queue import Queue

from robodk.robolink import Robolink, RUNMODE_SIMULATE, ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_FRAME, ITEM_TYPE_OBJECT
from robodk.robomath import rotx, roty, rotz

from robolink import *
from robodk import *
from robodk import robolink, robomath
from robodk.robolink import Robolink, ITEM_TYPE_ROBOT
#from robodk.robomath import eye
from robodk.robolink import *
from robodk.robomath import *

# This version works well damnit! The only problem is that if a collision occurs the whole thing breaks... That needs to be fixed.


## RoboDKMoveIt.py ##
RDK = Robolink()
robot = RDK.Item('UR3e', ITEM_TYPE_ROBOT)

np_queue = Queue()
camera_offset = np.array([-0.5, -0.21, 0.68]) # This is the position of the camera relative to the robot in meters. Measured by hand.

def is_within_workspace(position):
    """
    Check if the position is within the dexterous workspace of the robot.
    Return True if it is, False otherwise.
    """
    # Define workspace limits
    x_min, x_max = -400, 400
    y_min, y_max = -200, 400
    z_min, z_max = 120, 500

    # Check if the position is within the workspace
    if x_min <= position[0] <= x_max and y_min <= position[1] <= y_max and z_min <= position[2] <= z_max:
        return True
    else:
        return False
    
def create_target_point(position, robot):
    item_name = "Target Point"
    # Delete the old target point if it exists
    old_item = RDK.Item(item_name)
    if old_item.Valid():
        old_item.Delete()

    # Create a new target point
    target_point = RDK.AddTarget(item_name)

    # Set the position of the target point
    pose = np.eye(4)  # Create a 4x4 identity matrix
    pose[:3, 3] = position  # Set the translation components
    target_point.setPose(pose)

    # Disable collision checking between the target point and the robot
    RDK.setCollisionActivePair(False, target_point, robot)

    return target_point




def move_robot_to_position(position: np.ndarray):
    # Create a visual target point at the position
    create_target_point(position, robot)

    # Check if the target position is within the dexterous workspace
    if not is_within_workspace(position):
        print("Target position is not within the dexterous workspace of the robot. Aborting movement.")
        return

    # Hard-coded orientation (r=0, p=0, y=180)
    roll, pitch, yaw = np.radians(180), np.radians(0), np.radians(0)

    # Convert the orientation to a rotation matrix
    rx = rotx(roll)
    ry = roty(pitch)
    rz = rotz(yaw)
    rot_matrix = np.dot(np.dot(rz, ry), rx)

    # Combine the position and orientation to create a 4x4 homogeneous transformation matrix
    target_pose = np.eye(4)
    target_pose[:3, :3] = rot_matrix[:3, :3]
    target_pose[:3, 3] = position

    # Solve the inverse kinematics for the target pose to get all possible joint solutions
    target_joints_all = robot.SolveIK_All(target_pose)

    if target_joints_all is not None:
        # Iterate through the solutions and choose the first one with "Elbow Up" configuration
        for target_joints in target_joints_all:
            if target_joints[2] > 0:  # Check if the third joint is positive
                # Move the robot to the target joint values
                robot.MoveJ(target_joints)
                break
        else:
            print("Failed to find a solution with 'Elbow Up' configuration. Aborting movement.")
    else:
        print("Failed to calculate the target joint values. Aborting movement.")


      
""" def move_a2b(pointA: np.ndarray, pointB: np.ndarray, stop_event):  
    while not stop_event.is_set():  
        time.sleep(4)
        move_robot_to_position(pointA)
        time.sleep(4)
        move_robot_to_position(pointB) """


def move_a2b(pointA: np.ndarray, array_queue: Queue, stop_event):  
    while not stop_event.is_set():  
        time.sleep(8)
        move_robot_to_position(pointA)
        time.sleep(8)
        if not array_queue.empty():  # Check if the queue has any data
            new_array = array_queue.get()  # Get the last new_array from the queue
            if new_array.size > 0:  # Check if new_array contains any data
                pointB = new_array[-1][1]  # Get the 3D numpy array from the last item in new_array
                pointB[0] = pointB[0] * 1000 - 500
                pointB[1] = pointB[1] * 1000 - 210
                print("Moving to DetPoint: " + str(np.rint(pointB)))
                move_robot_to_position(np.rint(pointB))
        else:
            move_robot_to_position(pointA)

## Main ##

real_world_width = 0.7288
real_world_height = 0.4257
model = YOLO(r"C:\Users\grave\anaconda3\envs\monkeySort\monkeyExtraEnv\monkeySortSmall.pt")

predictions = []
stop_threads = False

def setup_pipeline():
    # Configure the Intel RealSense depth camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline
    profile = pipeline.start(config)

    # Get the depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, depth_scale

def detect_objects(pipeline, depth_scale):
    global stop_threads
    real_world_width = 1.29438
    real_world_height = 0.75607

    while not stop_threads:
        # Capture the depth and color frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())

        # Check if the depth frame is available
        if not depth_frame:
            continue

        # Convert the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        img_height, img_width, _ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixel_width = real_world_width / img_width
        pixel_height = real_world_height / img_height

        results = model.predict(img, conf=0.5, verbose=False) # Verbose=False stops printing into the terminal.

        # Clear the predictions list
        predictions.clear()

        for r in results:
            annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
                center = [(float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2]

                x = center[0] * pixel_width
                y = center[1] * pixel_height

                store_predictions(predictions, str(model.names[int(c)]), x, y)

                # Get the depth value at the center of the detected object
                depth_value = depth_image[int(center[1]), int(center[0])] * depth_scale

                # Draw a red dot at the center of the detected object
                dot_radius = 3
                dot_color = (0, 0, 255)  # BGR color for red
                dot_center = (int((float(b[0]) + float(b[2])) / 2), int((float(b[1]) + float(b[3])) / 2))
                cv2.circle(frame, dot_center, dot_radius, dot_color, -1)

                # Print the center of the predicted objects in pixels and meters
                #print(str(model.names[int(c)]) + " is located with center at in meters: " + str(center[0] * pixel_width) + " " + str(center[1] * pixel_height))

        frame = annotator.result()
        cv2.imshow('YOLO V8 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            stop_threads_event.set()
            wait_for_move_event.set()
            stop_threads = True
            break

    pipeline.stop()
    cv2.destroyAllWindows() 

def store_predictions(predictions, class_name, x, y):
    predictions.append([class_name, x, y])

def update_array_every_n_seconds(predictions, n=2.5):
    global stop_threads
    while not stop_threads:
        predictions.clear()
        time.sleep(n)

        # Define the data type for the structured numpy array
        dt = np.dtype([('label', np.unicode_, 16), ('data', np.float64, (3,))])
        new_array = np.empty(len(predictions), dtype=dt)
        
        for i, item in enumerate(predictions):
            class_name, x, y = item
            new_item = (class_name, [x, y, 140])  # The new data format
            new_array[i] = new_item

        # Put the new array into the queue
        np_queue.put(new_array)
        print("Updated numpy array:\n", new_array)


        
if __name__ == "__main__":
    bucket_point = np.array([390,0, 140])
    point2 = np.array([230, -190, 140])
    # Set up the pipeline and depth scale
    pipeline, depth_scale = setup_pipeline()

    stop_threads_event = threading.Event()
    wait_for_move_event = threading.Event()

    # Create a new thread for object detection
    detect_objects_thread = threading.Thread(target=detect_objects, args=(pipeline, depth_scale))
    update_array_every_n_seconds_thread = threading.Thread(target=update_array_every_n_seconds, args=(predictions,))
    move_a2b_thread = threading.Thread(target=move_a2b, args=(bucket_point, np_queue, stop_threads_event))
    #move_a2b_thread = threading.Thread(target=move_a2b, args=(bucket_point, point2, stop_threads_event))


    detect_objects_thread.start()
    update_array_every_n_seconds_thread.start()
    move_a2b_thread.start()

    detect_objects_thread.join()
    update_array_every_n_seconds_thread.join()