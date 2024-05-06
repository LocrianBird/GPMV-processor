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

def find_contour_tailored(cropped_image, center, average_radius):
    # Input parameters:
    #   1. Image data cropped around membrane
    #   2. Center - centroid of the membrane from cellpose
    #   3. Average radius - average radius of the membrane from cellpose
    # Output: list with points for contour of the membrane
    # 0. Define delta for angle dA
    # 1. Iterate over all angles [0, 2pi] with step dA
    # 2. Calculate rX and rY - points in image for radius and angle
    # 3. Iterate through all the points along the line at dA from the center to (rX,rY) with distance D = average_radius + 5px
    # 4. Find a point where luminosity/intensity changes with at least dL points (abruptly, meaning the biggest change)
    # 5. Add the point to the output array for contour, if found
    # 6. Return all found points as a list

    # Important definitions
    delta_angle = math.pi / 160
    radius_deviation = 10
    delta_luminosity = 8
    contour_points = []

    # Iterate over all angles from 0 to 2*pi with step delta_angle
    for angle in np.arange(0, 2 * math.pi, delta_angle):
        # Calculate the x and y coordinates for the radius start and endpoint
        start_radius = average_radius - radius_deviation
        end_radius = average_radius + radius_deviation

        start_x = int(center[0] + start_radius * math.cos(angle))
        start_y = int(center[1] + start_radius * math.sin(angle))
        end_x = int(center[0] + end_radius * math.cos(angle))
        end_y = int(center[1] + end_radius * math.sin(angle))

        # Ensure the radius does not go out of image bounds
        start_x = max(0, min(start_x, cropped_image.shape[1] - 1))
        start_y = max(0, min(start_y, cropped_image.shape[0] - 1))
        end_x = max(0, min(end_x, cropped_image.shape[1] - 1))
        end_y = max(0, min(end_y, cropped_image.shape[0] - 1))

        # Create a line iterator to get points between start and end points
        # line_iterator = cv2.lineIterator(cropped_image, (start_x, start_y), (end_x, end_y))
        line_x, line_y = line(start_x, start_y, end_x, end_y)

        # Initialize variables to find the maximum luminosity change
        max_change = 0
        best_point = None
        previous_value = None

        # Iterate through points along the line
        for x, y in zip(line_x, line_y):
            if 0 <= x < cropped_image.shape[1] and 0 <= y < cropped_image.shape[0]:
                value = cropped_image[y, x]
                print(value)
                if previous_value is not None:
                    luminosity_change = abs(int(value) - int(previous_value))
                    if luminosity_change > max_change and luminosity_change > delta_luminosity:
                        max_change = luminosity_change
                        best_point = (x, y)
                previous_value = value

        if best_point:
            contour_points.append(best_point)

    contour_points = np.array([contour_points], dtype=np.uint8).reshape(-1, 1, 2)
    return contour_points


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
    try:
        tailored_contour = find_contour_tailored(cropped_segment, center, averaged_radius)
    except:
        print("Tailored function crushed")
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
    cv2.drawContours(cropped_segment, [tailored_contour], -1, (0, 255, 0), thickness=1)
    cv2.imwrite(os.path.join(output_directory, f"{i}_tailored.tif"), image)
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
