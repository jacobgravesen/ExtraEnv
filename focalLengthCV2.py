import cv2
import numpy as np
import glob

# Define the chessboard dimensions (number of internal corners)
pattern_size = (9, 6)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
obj_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
object_points = []
image_points = []

# Get all the image files with the pattern
images = glob.glob(r'C:\Users\grave\anaconda3\envs\brownSort\ExtraEnv\cameraCallibrationFolder/*.png')

gray_shape = None

# Process each image
for image_file in images:
    # Load the image
    img = cv2.imread(image_file)
    
    # Check if the image is loaded
    if img is None:
        print(f"Failed to load {image_file}. Skipping...")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If corners are found, add object points and image points
    if ret:
        object_points.append(obj_points)
        image_points.append(corners)


    # Inside the loop, after finding the chessboard corners
    if ret:
        print(f"Found chessboard corners in {image_file}")
        object_points.append(obj_points)
        image_points.append(corners)
    else:
        print(f"Failed to find chessboard corners in {image_file}")


# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_shape, None, None)

# Extract the focal length
fx, fy = mtx[0, 0], mtx[1, 1]

print("Focal length (fx, fy):", fx, fy)