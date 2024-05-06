import glob
import json
import math
import os
import shutil
import sys
import time
from random import randint

import cv2
import numpy as np
import tifffile
from PIL import Image
from cellpose import models
from skimage.draw import line
from skimage import feature

# Read the image

# dirToStore = "C:/Users/mkana/Desktop/GPMV/GPMV_new/SingleTiffs/"
# # Check if the directory exists, if not, create it
# if not os.path.exists(dirToStore):
#     os.makedirs(dirToStore)
# else:
#     # Clear the directory if it's not empty
#     for file in os.listdir(dirToStore):
#         os.remove(os.path.join(dirToStore, file))
#
# # D:\GPMV Data
#
# for image_file in glob.glob(os.path.join("D:\\GPMV Data", "**\\*.tif"), recursive=True):
#     imageAll = tifffile.imread(image_file)
#     lengthOfStack = imageAll.shape[0]  # Get the length of the stack
#     # for index in range(lengthOfStack):  # Loop through each frame
#     frame = randint(0, lengthOfStack)
#     image = np.array(imageAll[frame, :, :], dtype=np.uint16)
#     normalized = (image - image.min()) * 255.0 / (image.max() - image.min())
#     image = Image.fromarray(normalized.astype(np.uint8), mode='L')
#     image_name = f"{os.path.basename(os.path.dirname(image_file))}_{os.path.basename(image_file).removesuffix('.tif')}_{frame}"
#     data = list(image.getdata())
#     if data.count(0) / float(len(data)) >= 0.5 or data.count(255) / float(len(data)) >= 0.5:
#         print(f"{image_name} is invalid")
#         continue
#     image.save(os.path.join(dirToStore, f"{image_name}.tif"))
#
# print("Finished processing")

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to clear the contents of a directory
def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

# Function to contrast stretch images
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = ((image - min_val) / (max_val - min_val)) * 255
    return stretched_image.astype(np.uint8)

def smooth_contour(points, window_size=5):
    if len(points) < window_size:
        return points
    smoothed_points = []
    for i in range(len(points)):
        avg_x = np.mean([point[0] for point in points[max(0, i-window_size//2):min(len(points), i+window_size//2+1)]])
        avg_y = np.mean([point[1] for point in points[max(0, i-window_size//2):min(len(points), i+window_size//2+1)]])
        smoothed_points.append((int(avg_x), int(avg_y)))
    return smoothed_points

import numpy as np
import math
from skimage.draw import line

import numpy as np
import math
from skimage.draw import line

def find_contour_tailored(cropped_image, center, average_radius, radius_deviation=10, delta_brightness=15, window_size=5):
    """
    Find the contour points around a given center in a cropped image by detecting the most significant increase in luminosity,
    including a smoothing mechanism to reduce noise.

    Parameters:
        cropped_image (np.array): The cropped image around the region of interest.
        center (tuple): The (x, y) coordinates of the center of the region.
        average_radius (int): The average radius from the center to search for the contour.
        radius_deviation (int): The deviation from the average radius to define the search range.
        delta_brightness (int): The threshold for detecting a significant luminosity change.
        window_size (int): The number of points on either side of the current point to consider for averaging.

    Returns:
        np.array: The array of points forming the detected contour.
    """
    delta_angle = 1/average_radius**2  # Angle step size
    contour_points = []

    start_radius = average_radius - radius_deviation
    end_radius = average_radius + radius_deviation

    for angle in np.arange(0, 2 * math.pi, delta_angle):
        start_x = int(center[0] + start_radius * math.cos(angle))
        start_y = int(center[1] + start_radius * math.sin(angle))
        end_x = int(center[0] + end_radius * math.cos(angle))
        end_y = int(center[1] + end_radius * math.sin(angle))

        # Ensure the coordinates are within image bounds
        start_x, start_y = np.clip([start_x, start_y], 0, np.array(cropped_image.shape[1::-1]) - 1)
        end_x, end_y = np.clip([end_x, end_y], 0, np.array(cropped_image.shape[1::-1]) - 1)

        line_x, line_y = line(start_x, start_y, end_x, end_y)
        values = cropped_image[line_y, line_x]

        # Apply smoothing to the brightness values along the line
        smoothed_values = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        best_point = None
        max_increase = 0

        for i in range(1, len(smoothed_values)):
            brightness_change = smoothed_values[i] - smoothed_values[i - 1]
            # Look for the largest increase from dark to bright
            if brightness_change > max_increase or brightness_change > delta_brightness:
                max_increase = brightness_change
                idx = i + window_size // 2  # Adjust index for the window
                best_point = (line_x[idx], line_y[idx])

        if best_point:
            contour_points.append(best_point)

    return np.array([contour_points], dtype=np.int32) if contour_points else np.array([], dtype=np.int32).reshape(-1, 1, 2)




# D:\GPMV Data\19.03.2024\CaSki P15\_s1_44.tif
bounding_box = None
model = models.Cellpose(gpu=True, model_type='cyto')

def cellpose(image):
    global model
    masks, _, _, _ = model.eval(image, diameter=None, channels=[0, 0])
    print(f"Found {len(masks)}")
    contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or len(contours) == 0:
        print("No contours")
        return

    max_contour = None
    contours = list(contours)
    contours.sort(key=cv2.contourArea, reverse=True)
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        radii = [math.dist(center, p[0]) for p in contour]
        averaged_radius = np.mean(radii)
        if averaged_radius > 70:
            print("Too big")
            continue
        elif averaged_radius < 30:
            print("Too small")
            break

        undulations = [radius - averaged_radius for radius in radii]
        abs_undulations = [abs(un) for un in undulations]
        if max(abs_undulations) <= 5:
            print(f"Too variable")
            max_contour = contour
            break

    if max_contour is not None:
        print(f"Adequate contour found")
    else:
        print(f"No adequate contour found")
        return

    # Find bounding rectangle around the biggest segment
    bounding_box = cv2.boundingRect(max_contour)
    # Add 5 pixels to each side of the bounding rectangle
    bounding_box = list(bounding_box)
    bounding_box[0] -= 20
    bounding_box[1] -= 20
    bounding_box[2] += 40
    bounding_box[3] += 40

    return max_contour, bounding_box, averaged_radius, undulations, center


output_directory = os.path.join(os.getcwd(), "Segmented Membranes")
create_directory(output_directory)
clear_directory(output_directory)
for i, image in enumerate(tifffile.imread("D:/GPMV Data/19.03.2024/CaSki P15/_s1_44.tif")):
    image = ((image - image.min()) * 255.0 / (image.max() - image.min())).astype(np.uint8)
    if bounding_box is None:
        ret = cellpose(image)
        if ret:
            max_contour, bounding_box, averaged_radius, undulations, center = ret
        else:
            print("Couldn't find counter for bounding box")
            continue


        # Crop the segmented image using the bounding rectangle
    cropped_segment = image[bounding_box[1]:bounding_box[1]+bounding_box[3]+1, bounding_box[0]:bounding_box[0]+bounding_box[2]+1]
    adjusted_center = (center[0] - bounding_box[0], center[1] - bounding_box[1])
    tailored_contour = find_contour_tailored(cropped_segment, adjusted_center, averaged_radius)
    image = image
    print(len(tailored_contour))
    try:  # Check if contour is not empty
        cv2.drawContours(cropped_segment, [tailored_contour], -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(output_directory, f"{i}_tailored.png"), image)
    except Exception as e:
        print(f"No tailored contour found wit {e}")
    #cropped_segment = cv2.Sobel(src=cropped_segment, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=7)
    #cropped_segment = np.vectorize(lambda x: 255 if x else 0)(feature.canny(cropped_segment, sigma=3)).astype(np.uint8)
    ret = cellpose(cropped_segment)
    if ret:
        new_contour, nb, averaged_radius, undulations, _ = ret
    else:
        print("Couldn't find counter with bounding box")
        continue


    #bounding_box[0] = round(center[0]) + nb[0]
    #bounding_box[1] = round(center[1]) + nb[1]
    # image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cropped_segment = image2[bounding_box[1]:bounding_box[1]+bounding_box[3]+1, bounding_box[0]:bounding_box[0]+bounding_box[2]+1]
    # cv2.drawContours(image, [max_contour], -1, (0, 0, 0), thickness=1)
    cv2.drawContours(cropped_segment, [new_contour], -1, (0, 0, 0), thickness=1)
    cv2.imwrite(os.path.join(output_directory, f"{i}.tif"), image)
    # cv2.imwrite("2.png", image2)


sys.exit(0)

model = models.Cellpose(gpu=True, model_type='cyto')
input_directory = 'C:/Users/mkana/Desktop/GPMV/GPMV_new/SingleTiffs/'
output_directory = os.path.join(os.getcwd(), "Segmented Membranes")
create_directory(output_directory)
clear_directory(output_directory)
log_file_path = os.path.join(output_directory, "Hela_P7_10.json")

data_for_all_frames = []
counter = 1
diameters = []  # List to store diameters
frame_start, frame_end = 0.0, 0.0
time_counter = True

for filename in sorted(os.listdir(input_directory)):
    if filename.endswith(".tif"):
        image_path = os.path.join(input_directory, filename)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        image = cv2.equalizeHist(img)



        # # apply close morphology
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)

        # threshold to binary
        #image = contrast_stretching(image)
        # cv2.imshow('ehee', contrasted_image)
        # cv2.waitKey(0)

        if time_counter:
            frame_start = time.time()

        masks, _, _, _ = model.eval(image, diameter=None, channels=[0, 0])

        if time_counter:
            time_counter = False
            frame_end = time.time()
            total_seconds = frame_end - frame_start
            minutes, seconds = divmod(total_seconds, 60)
            print(f'Frame elapsed time = {int(minutes)}m {seconds:.5f}s')

        contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours) == 0:
            continue

        max_contour = None
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        for contour in contours:
            center, _ = cv2.minEnclosingCircle(contour)
            radii = [math.dist(center, p[0]) for p in contour]
            averaged_radius = np.mean(radii)
            if averaged_radius > 70:
                print("Too big")
                continue
            elif averaged_radius < 30:
                print("Too small")
                break

            undulations = [radius - averaged_radius for radius in radii]
            abs_undulations = [abs(un) for un in undulations]
            if max(abs_undulations) <= 5:
                print(f"Too variable - {filename}")
                max_contour = contour
                break

        if max_contour is not None:
            print(f"Adequate contour found - {filename}")
        else:
            print(f"No adequate contour found - {filename}")
            continue

        (x_circle, y_circle), radius = cv2.minEnclosingCircle(max_contour)
        diameter = (radius * 2)
        diameters.append(diameter)  # Append diameter to the list
        M = cv2.moments(max_contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        frame_data = {
            "frame_number": counter,
            "centroid": [center_x, center_y],
            "circumference_coordinates": max_contour.reshape(-1, 2).tolist(),
            "diameter": diameter
        }
        data_for_all_frames.append(frame_data)

        # Image processing and logging
        circumference_image = np.zeros_like(image)
        cv2.drawContours(img, [max_contour], -1, (0, 0, 0), thickness=2)
        x, y, w, h = cv2.boundingRect(max_contour)
        side_length = max(w, h)
        x_square = max(0, center_x - 75)
        y_square = max(0, center_y - 75)
        side_length = min(min(image.shape[0] - y_square, image.shape[1] - x_square), 150)
        cropped_circumference = circumference_image[y_square:y_square + side_length, x_square:x_square + side_length]

        cv2.imwrite(os.path.join(output_directory, filename), img)
        counter += 1

# Write the collected data to a JSON file
# with open(log_file_path, 'w') as log_file:
#     json.dump(data_for_all_frames, log_file)

# Calculate the average diameter from the 'diameters' list and convert to microns
average_diameter_microns = np.mean(diameters)*0.16
print(f"Average Diameter: {average_diameter_microns:.3f} microns")
