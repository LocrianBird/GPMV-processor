import glob
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from random import randint
from typing import Any, Optional

import PIL.Image
import cv2
import numpy as np
import tifffile
from PIL import Image, ImageFilter, ImageEnhance
from cellpose import models
from skimage.draw import line_aa
from skimage import filters
import itertools
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from numpy import convolve, ones
from skimage.measure import find_contours
import sympy as sy
import sympy.geometry as gm


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
        avg_x = np.mean(
            [point[0] for point in points[max(0, i - window_size // 2):min(len(points), i + window_size // 2 + 1)]])
        avg_y = np.mean(
            [point[1] for point in points[max(0, i - window_size // 2):min(len(points), i + window_size // 2 + 1)]])
        smoothed_points.append((int(avg_x), int(avg_y)))
    return smoothed_points


import numpy as np
import math
from skimage.draw import line

import numpy as np
import math
from skimage.draw import line

model = models.Cellpose(gpu=True, model_type='cyto')

def adjust_contour_to_membrane(image: np.ndarray, contour_coords: np.ndarray, centroid: tuple[int, int]) -> np.ndarray:
    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    # return
    def seg_intersect(a1, a2, b1, b2):
        start_x = max(min(a1[0], a2[0]), min(b1[0], b2[0]))
        end_x = min(max(a1[0], a2[0]), max(b1[0], b2[0]))
        start_y = max(min(a1[1], a2[1]), min(b1[1], b2[1]))
        end_y = min(max(a1[1], a2[1]), max(b1[1], b2[1]))
        if start_x > end_x or start_y > end_y:
            return None

        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        out = (num / denom.astype(float)) * db + b1
        return None if any(map(lambda x: math.isnan(x), out)) else out if start_x <= out[0] <= end_x and start_y <= out[1] <= end_y else None

    """
    :param image:
    :param contour_coords: shape (1, 1, 2)
    :param centroid:
    :return:
    """

    out = []
    offset = 10
    start_radius = 75
    delta_angle = math.radians(30)
    box_length = 20

    for angle in np.arange(0, 2 * math.pi, math.radians(10)):
        start_x = round(centroid[0] + start_radius * math.cos(angle))  # rename later
        start_y = round(centroid[1] + start_radius * math.sin(angle))  # rename later :c
        inter_point = None
        for start, end in zip(contour_coords, contour_coords[1:]):
            inter_point = seg_intersect(np.array(centroid), np.array([start_x, start_y]), start[0], end[0])
            if inter_point is not None:
                break

        if inter_point is not None:
            inter_radius = math.dist(centroid, inter_point)
            inter_radius -= offset
            left_line_start_x = round(centroid[0] + inter_radius * math.cos(angle-delta_angle))
            left_line_start_y = round(centroid[1] + inter_radius * math.sin(angle-delta_angle))
            right_line_start_x = round(centroid[0] + inter_radius * math.cos(angle+delta_angle))
            right_line_start_y = round(centroid[1] + inter_radius * math.sin(angle+delta_angle))
            abs_end_x = round(box_length*math.cos(angle))
            abs_end_y = round(box_length*math.sin(angle))
            left_line_end_x = left_line_start_x + abs_end_x
            left_line_end_y = left_line_start_y + abs_end_y
            right_line_end_x = right_line_start_x + abs_end_x
            right_line_end_y = right_line_start_y + abs_end_y
            left_line = list(zip(*line(left_line_start_x, left_line_start_y, left_line_end_x, left_line_end_y)))
            right_line = list(zip(*line(right_line_start_x, right_line_start_y, right_line_end_x, right_line_end_y)))
            s = left_line + right_line
            all_xs = list(map(lambda x: x[0], s))
            all_ys = list(map(lambda x: x[1], s))
            min_x, max_x = min(all_xs), max(all_xs)
            min_y, max_y = min(all_ys), max(all_ys)

            last_value = None
            candidate = None
            last_pair = None
            checked = 0
            for idx, (left_point, right_point) in enumerate(zip(left_line, right_line)):
                cross = np.max(image[*map(lambda x: np.array(x), line(*left_point, *right_point))])

                if last_value is not None:
                    if cross - last_value > 20:
                        if candidate is None:
                            candidate = last_pair

                        if checked == 5:
                            break

                        checked += 1

                    elif candidate is not None:
                        checked = 0
                        last_value = cross
                        candidate = None
                    else:
                        last_value = cross
                        last_pair = (left_point, right_point)
                else:
                    last_value = cross
                    last_pair = (left_point, right_point)

            if candidate is not None:
                inter_point = seg_intersect(np.array(centroid), np.array([start_x, start_y]), np.array(candidate[0]), np.array(candidate[1]))
                if inter_point is not None:
                    out.append(inter_point)
    return None if len(out) == 0 else np.array(out).reshape((-1, 1, 2)).astype(np.int32)


def find_contour_tailored(cropped_image,
                          center,
                          average_radius,
                          radius_deviation=8,
                          delta_brightness=20,
                          window_size=5):
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
    delta_angle = 1 / average_radius ** 2  # Angle step size
    contour_points = []

    start_radius = average_radius - radius_deviation
    end_radius = average_radius + radius_deviation

    for angle in np.arange(0, 2 * math.pi, delta_angle):
        start_x = round(center[0] + start_radius * math.cos(angle))
        start_y = round(center[1] + start_radius * math.sin(angle))
        end_x = round(center[0] + end_radius * math.cos(angle))
        end_y = round(center[1] + end_radius * math.sin(angle))

        # Ensure the coordinates are within image bounds
        start_x, start_y = np.clip([start_x, start_y], 0, np.array(cropped_image.shape[1::-1]) - 1)
        end_x, end_y = np.clip([end_x, end_y], 0, np.array(cropped_image.shape[1::-1]) - 1)

        line_x, line_y, _ = line_aa(start_x, start_y, end_x, end_y)
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
                idx = i + window_size // 2
                best_point = (line_x[idx], line_y[idx])

        if best_point:
            contour_points.append(best_point)

    return np.array([contour_points], dtype=np.int32) if contour_points else np.array([], dtype=np.int32).reshape(-1, 1,
                                                                                                                  2)


@dataclass
class Image:
    name: str
    data: np.ndarray


@dataclass
class Iteration:
    input_file: str
    output_dir: str
    image: Optional[Image]
    edited_images: list[Image] = field(default_factory=lambda: [])


@dataclass
class ContourData:
    centroid: tuple[int, int]
    averaged_radius: float
    contour_coords: list


@dataclass
class IterationOrdered:
    input_file: str
    output_dir: str
    image: Optional[Image]
    contour_data: Optional[ContourData]



def read_file_random_frame(iteration: Iteration) -> Iteration:
    image_all = tifffile.imread(iteration.input_file)
    length_of_stack = image_all.shape[0]  # Get the length of the stack
    # for index in range(length_of_stack):  # Loop through each frame
    frame = randint(0, length_of_stack - 1)
    image = np.array(image_all[frame, :, :], dtype=np.uint16)
    normalized = (image - image.min()) * 255.0 / (image.max() - image.min())
    image_name = f"{os.path.basename(os.path.dirname(iteration.input_file))}_{os.path.basename(iteration.input_file).removesuffix('.tif')}_{frame}"
    iteration.image = Image(name=image_name, data=normalized.astype(np.uint8))
    return iteration

from scipy.interpolate import splprep, splev


def is_valid(iteration_ordered: IterationOrdered) -> bool:
    data = list(iteration_ordered.image.data.flatten())
    return not ((data.count(0) / float(len(data)) >= 0.5) or (data.count(255) / float(len(data)) >= 0.5))

def find_max_valid_contour_ordered(contours: tuple[list, ...]) -> Optional[ContourData]:
    max_contour = None
    contours = list(contours)
    contours.sort(key=cv2.contourArea, reverse=True)
    for contour in contours:
        # peri = cv2.arcLength(contour, True)
        # contour = cv2.approxPolyDP(contour, 0.0065*peri, True)
        # Convert contour to a numpy
        if len(contour) < 5:
            print("Not enough points on a contour")
            continue

        # image = np.zeros((512, 512), dtype=np.uint8)
        # cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=1)
        # cv2.imshow("", image)
        # cv2.waitKey()

        # temp = contour.reshape(-1, 2)
        # tck, u = splprep([temp[:,0], temp[:,1]], s=35)
        #
        # # Evaluate spline on the parametric range
        # new_points = splev(np.linspace(0, 1, 100), tck)
        #
        # # Draw the smooth curve
        # contour = np.array(new_points).T.reshape(-1, 1, 2).astype(np.int32)
        # image = np.zeros((512, 512), dtype=np.uint8)
        # cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=1)
        # cv2.imshow("", image)
        # cv2.waitKey()

        center, _ = cv2.minEnclosingCircle(contour)
        radii = [math.dist(center, p[0]) for p in contour]
        averaged_radius = np.mean(radii)


        # area = cv2.contourArea(contour)
        # perimeter = cv2.arcLength(contour, True)
        # if perimeter == 0:
        #     continue  # Avoid division by zero
        # circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # if 0.95 >= circularity >= 1.05:
        #     print(f"Not circular {circularity}")]
        #     continue
        ellipse = cv2.fitEllipse(contour)
        ellipse_contour = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                           (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                                           int(ellipse[2]), 0, 360, 1)
        matching = cv2.matchShapes(contour, ellipse_contour, 1, 0.0)
        if matching > 0.2:
            print(f"Not circular {matching}")
            continue
        elif averaged_radius > 70:
            print("Too big")
            continue
        elif averaged_radius < 30:
            print("Too small")
            break
        # elif not cv2.isContourConvex(contour):
        #     print("Not convex")
        #     continue

        undulations = [radius - averaged_radius for radius in radii]
        abs_undulations = [abs(un) for un in undulations]
        if max(abs_undulations) <= 10:
            max_contour = contour
            break

    if max_contour is not None:

        # window_length = 7
        #
        # # Apply Gaussian Filter
        # smoothed_x = gaussian_filter(max_contour[..., 0, 0], sigma=window_length // 6, mode='wrap')
        # smoothed_y = gaussian_filter(max_contour[..., 0, 1], sigma=window_length // 6, mode='wrap')
        #
        # # Reshape back to the original shape (1, 1, 2)
        # max_contour = np.round(np.stack((smoothed_x, smoothed_y), axis=-1)).astype(np.int32).reshape(-1, 1, 2)
        contour_data = ContourData((round(center[0]), round(center[1])), averaged_radius, max_contour)
        print(f"Adequate contour found")
        return contour_data
    else:
        print(f"No adequate contour found")
        return None


def preprocess_in_order(iteration_ordered: IterationOrdered) -> IterationOrdered:

    def preprocess_step(iteration_ordered: IterationOrdered) -> Optional[IterationOrdered]:
        global model
        ksizes = range(1, 14, 2)
        sigmaXs = range(0, 7)
        blockSizes = range(3, 30, 2)
        cs = range(1, 7)
        combination = list(itertools.product(ksizes, sigmaXs, blockSizes, cs))
        combination.insert(0, (9, 0, 21, 1))
        if iteration_ordered.contour_data is None:
            masks, _, _, _ = model.eval(iteration_ordered.image.data, diameter=None, channels=[0, 0])
            masks = np.vectorize(lambda x: 0.0 if x > 0 else 1.0)(masks).astype(np.double)
            cnts = find_contours(masks)
            # blur = cv2.GaussianBlur(iteration_ordered.image.data, (ksize, ksize), sigmaX)
            # thresh_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
            # # if '16' in iteration_ordered.input_file:
            # #     cv2.imshow("", thresh_image)
            # #     cv2.waitKey()
            # cnts = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in cnts:
                for idx, (y, x) in enumerate(contour):
                    contour[idx] = np.array((x, y))
            cnts = tuple(map(lambda x: np.round(x).reshape((-1, 1, 2)).astype(np.int32), cnts))
            # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            max_cnt = find_max_valid_contour_ordered(cnts)
            if max_cnt is not None:
                # max_cnt.contour_coords = adjust_contour_to_membrane(iteration_ordered.image.data, max_cnt.contour_coords, cv2.minEnclosingCircle(max_cnt.contour_coords)[0])
                # max_cnt.centroid = cv2.minEnclosingCircle(max_cnt.contour_coords)[0]
                # max_cnt.averaged_radius = np.mean([math.dist(max_cnt.centroid, p[0]) for p in max_cnt.contour_coords])

                bounding_box = cv2.boundingRect(max_cnt.contour_coords)
                # Add [side] pixels to each side of the bounding rectangle
                bounding_box = list(bounding_box)
                side = 10
                bounding_box[0] -= side
                bounding_box[1] -= side
                bounding_box[2] += side * 2
                bounding_box[3] += side * 2

                factor = 32
                cutout = np.zeros((bounding_box[3] * factor, bounding_box[2] * factor), dtype=np.uint8)
                for x in range(bounding_box[0], bounding_box[0] + bounding_box[2]):
                    for y in range(bounding_box[1], bounding_box[1] + bounding_box[3]):
                        sx, sy = (x - bounding_box[0]) * factor, (y - bounding_box[1]) * factor
                        color = iteration_ordered.image.data[y, x]

                        for nx in range(sx, sx + factor):
                            for ny in range(sy, sy + factor):
                                cutout[ny, nx] = color

                        cv2.putText(cutout, str(color), (sx, sy + factor // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0))

                cv2.imwrite("output.png", cutout)

                iteration_ordered.contour_data = max_cnt
                cv2.drawContours(iteration_ordered.image.data, [max_cnt.contour_coords], -1, (0, 0, 0), 1)
            else:
                return None
        else:
            bounding_box = cv2.boundingRect(iteration_ordered.contour_data.contour_coords)
            # Add [side] pixels to each side of the bounding rectangle
            bounding_box = list(bounding_box)
            side = 10
            bounding_box[0] -= side
            bounding_box[1] -= side
            bounding_box[2] += side * 2
            bounding_box[3] += side * 2

            cropped_segment = iteration_ordered.image.data[bounding_box[1]:bounding_box[1] + bounding_box[3] + 1,
                              bounding_box[0]:bounding_box[0] + bounding_box[2] + 1]

            for ksize, sigmaX, block_size, c in combination:
                blur = cv2.GaussianBlur(cropped_segment, (ksize, ksize), sigmaX)
                thresh_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size,
                                                     c)
                cnts = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                max_cnt = find_max_valid_contour_ordered(cnts)
                if max_cnt is not None:
                    max_cnt.centroid = (bounding_box[0] + max_cnt.centroid[0], bounding_box[1] + max_cnt.centroid[1])
                    for coordinate in max_cnt.contour_coords:
                        coordinate[0][0] += bounding_box[0]
                        coordinate[0][1] += bounding_box[1]
                    iteration_ordered.contour_data = max_cnt
                    cv2.drawContours(iteration_ordered.image.data, [max_cnt.contour_coords], -1, (0, 0, 0), 1)
                    break

        return iteration_ordered

    return preprocess_step(iteration_ordered) if is_valid(iteration_ordered) else None


def process_in_order(iteration: Iteration):
    pipe = [preprocess_in_order, output_in_order]
    inp = IterationOrdered(iteration.input_file, iteration.output_dir, None, None)
    for index, frame in enumerate(tifffile.imread(iteration.input_file)):
        normalized = ((frame - frame.min()) * 255.0 / (frame.max() - frame.min())).astype(np.uint8)
        data = list(normalized.flatten())
        if sum(1 for x in data if x < 10 or x > 245) / len(data) >= 0.5:
            print(f"Input file {iteration.input_file} is broken")
            break
        image_name: str = f"{os.path.basename(os.path.dirname(iteration.input_file))}_{os.path.basename(iteration.input_file).removesuffix('.tif')}_{index}"
        inp.image = Image(image_name, normalized)
        for idx, fun in enumerate(pipe):
            inp = fun(inp)
            if idx < len(pipe) - 1 and inp is None:
                print(f"Image {iteration.input_file} is invalid")
                break
        break


def output_in_order(iteration_ordered: IterationOrdered) -> IterationOrdered:
    PIL.Image.fromarray(iteration_ordered.image.data, mode='L').save(os.path.join(iteration_ordered.output_dir, f"{iteration_ordered.image.name}.png"))
    return iteration_ordered


def read(path: str, output_directory: str, pipe: list):
    counter = 0
    clear_directory(output_directory)
    image_files = [path] if os.path.isfile(path) else glob.glob(os.path.join(path, "**\\*.tif"), recursive=True)
    for image_file in image_files:
        inp = Iteration(image_file, output_directory, None)
        for idx, fun in enumerate(pipe):
            inp = fun(inp)
            if idx < len(pipe) - 1 and inp is None:
                counter += 1
                print(f"Image {image_file} is invalid")
                break
    print(f"{float(counter)/len(image_files)*100:.2f}%")


def find_max_valid_contour(contours: list[list]) -> Optional[list]:
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
            max_contour = contour
            break

    if max_contour is not None:
        print(f"Adequate contour found")
        return max_contour
    else:
        print(f"No adequate contour found")
        return None


def preprocess(iteration: Iteration) -> Optional[Iteration]:
    def is_valid(iteration: Iteration) -> bool:
        data = list(iteration.image.data.flatten())
        return not ((data.count(0) / float(len(data)) >= 0.5) or (data.count(255) / float(len(data)) >= 0.5))

    def preprocess_step(iteration: Iteration) -> Optional[Iteration]:
        ksizes = range(1, 14, 2)
        sigmaXs = range(0, 7)
        blockSizes = range(9, 30, 2)
        cs = range(1, 7)
        edited_images: list[Image] = []
        for k, sigmaX, blockSize, c in [[9, 0, 21, 1]]:
            blur = cv2.GaussianBlur(iteration.image.data, (k, k), sigmaX)
            image_name: str = f"{iteration.image.name}+_{k}_{sigmaX}_{blockSize}_{c}"
            thresh_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, c)
            image_copy = iteration.image.data.copy()
            cnts = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            max_cnt = find_max_valid_contour(cnts)
            if max_cnt is not None:

                cv2.drawContours(image_copy, [max_cnt], -1, (0, 0, 0), 1)
            else:
                return None
            edited_images.append(Image(image_name, image_copy))

        return Iteration(iteration.input_file, iteration.output_dir, iteration.image, edited_images)

    return preprocess_step(iteration) if is_valid(iteration) else None


def output(iteration: Iteration) -> None:
    for image in iteration.edited_images:
        PIL.Image.fromarray(image.data, mode='L').save(os.path.join(iteration.output_dir, f"{image.name}.png"))


# pipe = [
#     read_random,
#     preprocess,
#     output
# ]
# execute(pipe, initial_data)

read('D:/GPMV Data',
            'C:/Users/mkana/Desktop/GPMV/GPMV_new/Segmented Membranes',
     [process_in_order])
# output(preprocess(read_random("D:/GPMV Data")), "C:/Users/mkana/Desktop/GPMV/GPMV_new/Segmented Membranes")

sys.exit(0)

# D:\GPMV Data\19.03.2024\CaSki P15\_s1_44.tif
bounding_box = None



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
    cropped_segment = image[bounding_box[1]:bounding_box[1] + bounding_box[3] + 1,
                      bounding_box[0]:bounding_box[0] + bounding_box[2] + 1]
    adjusted_center = (center[0] - bounding_box[0], center[1] - bounding_box[1])
    tailored_contour = find_contour_tailored(cropped_segment,
                                             adjusted_center,
                                             averaged_radius,
                                             10,
                                             5,
                                             5)
    print(len(tailored_contour))
    try:  # Check if contour is not empty
        cv2.drawContours(cropped_segment, [tailored_contour], -1, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(output_directory, f"{i}_tailored.png"), image)
    except Exception as e:
        print(f"No tailored contour found wit {e}")
    # cropped_segment = cv2.Sobel(src=cropped_segment, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=7)
    # cropped_segment = np.vectorize(lambda x: 255 if x else 0)(feature.canny(cropped_segment, sigma=3)).astype(np.uint8)
    ret = cellpose(cropped_segment)
    if ret:
        new_contour, nb, averaged_radius, undulations, _ = ret
    else:
        print("Couldn't find counter with bounding box")
        continue

    # bounding_box[0] = round(center[0]) + nb[0]
    # bounding_box[1] = round(center[1]) + nb[1]
    # image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cropped_segment = image2[bounding_box[1]:bounding_box[1]+bounding_box[3]+1, bounding_box[0]:bounding_box[0]+bounding_box[2]+1]
    # cv2.drawContours(image, [max_contour], -1, (0, 0, 0), thickness=1)
    cv2.drawContours(cropped_segment, [new_contour], -1, (0, 0, 0), thickness=1)
    cv2.imwrite(os.path.join(output_directory, f"{i}.png"), image)
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
        # image = contrast_stretching(image)
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
average_diameter_microns = np.mean(diameters) * 0.16
print(f"Average Diameter: {average_diameter_microns:.3f} microns")
