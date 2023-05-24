import math
import cv2
import numpy as np
import pyrealsense2 as rs

# This script provides the calculations for how much of the real world the camera is observing.
# First, run the script and it will provide the length from the camera to the surface in the center of the image.
# This will be printed in the terminal.
# Apply this distance (in meters) as well as FOV_H and FOV_V specs for your specific camera.

# specs for intel 435i camera
FOV_H, FOV_V = 87, 58 # H = horizontal, V = vertial
#camera_height_m = 0.682 # this is the height the camera is from the table in. It can be calculated using the distance sensor.
camera_height_m = 0.384

HFOV_rad = math.radians(FOV_H)  # Convert HFOV to radians
VFOV_rad = math.radians(FOV_V)  # Convert VFOV to radians

visible_width_m = 2 * camera_height_m * math.tan(HFOV_rad / 2)
visible_height_m = 2 * camera_height_m * math.tan(VFOV_rad / 2)

print("visible width: " +str(visible_width_m))
print("visible height: " + str(visible_height_m))

# Configure the Intel RealSense depth camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get the depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

while True:
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

    center_pixel_coordinates = (img_width // 2, img_height // 2)

    # Access the depth value at the center pixel coordinates
    center_depth_value = depth_image[center_pixel_coordinates[1], center_pixel_coordinates[0]] * depth_scale
    # Print the center depth value
    print(f"Depth at the center of the image: {center_depth_value:.3f} meters")

    # Draw a blue dot at the center of the image
    dot_radius = 3
    dot_color = (255, 0, 0)  # BGR color for blue
    cv2.circle(frame, center_pixel_coordinates, dot_radius, dot_color, -1)

    cv2.imshow('Center Depth Visualization', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

pipeline.stop()
cv2.destroyAllWindows()