#   ROBT310:    Project – Final Report
#   Name:       Dias Manap,     Moldir Berkaliyeva,     Zhanel Yessirkepbaeva
#   Student ID: 201871578,      201730450,              201824656

import numpy as np
import cv2
import matplotlib
from scipy.interpolate import UnivariateSpline
from enum import Enum

# simulation variables

width = 480
height = 640

class Type(Enum):
    PROTANOMALY = 1
    PROTANOPIA = 2
    DEUTERANOMALY = 3
    DEUTERANOPIA = 4
    TRITANOMALY = 5
    TRITANOPIA = 6
    ACHROMATOMALY = 7
    MONOCHROMACY = 8

# simulation values for each type of color blindness
simulation_matrix_map = {
	Type.PROTANOPIA.name:
		[[0.567, 0.433, 0],
		[0.558, 0.442, 0],
		[0, 0.242, 0.758]],
	Type.PROTANOMALY.name:
		[[0.817, 0.183, 0],
		[0.333, 0.667, 0],
		[0, 0.125, 0.875]],
	Type.DEUTERANOPIA.name:
		[[0.625, 0.375, 0],
		[0.7, 0.3, 0],
		[0, 0.3, 0.7]],
	Type.DEUTERANOMALY.name:
		[[0.8, 0.2, 0],
		[0.258, 0.742, 0],
		[0,0.142,0.858]],
	Type.TRITANOPIA.name:
		[[0.95, 0.05, 0],
		[0, 0.433, 0.567],
		[0, 0.475, 0.525]],
	Type.TRITANOMALY.name:
		[[0.967, 0.033, 0],
		[0, 0.733, 0.267],
		[0, 0.183, 0.817]],
	Type.MONOCHROMACY.name:
		[[0.299, 0.587, 0.114],
		[0.299, 0.587, 0.114],
		[0.299, 0.587, 0.114]],
	Type.ACHROMATOMALY.name:
		[[0.618, 0.320, 0.062],
		[0.163, 0.775, 0.062],
		[0.163, 0.320, 0.516]]
}

def simulate(rgb_image, current_type):
    simulation_matrix = simulation_matrix_map[current_type.name]
    simulation_matrix = np.array(simulation_matrix)
    flat_rgb_image = rgb_image.reshape(1, 640*480, 3)
    flat_rgb_image = flat_rgb_image[0]
    flat_transformed = np.einsum('ij,kj->ki', simulation_matrix, flat_rgb_image)
    flat_transformed = flat_transformed.astype(int)
    flat_transformed = np.array(flat_transformed)
    new_rgb_image = flat_transformed.reshape(width, height, 3)
    return new_rgb_image


def correct_opia(rgb_image,hue_value):
	flat_rgb_image = np.array(rgb_image.reshape(1,width*height,3)[0], dtype="float")
	flat_rgb_image = np.divide(flat_rgb_image, 255)
	hsv_image = matplotlib.colors.rgb_to_hsv(flat_rgb_image)
	
	hsv_image[:,0]+=hue_value
	hsv_image[hsv_image > 1] -= 1
	hsv_image[hsv_image < 0] = 0

	temp_rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)
	new_rgb_image = np.multiply(temp_rgb_image, 255)
	
	new_rgb_image = new_rgb_image.reshape(width, height, 3).astype(int)
	return new_rgb_image

def correct_omaly(rgb_image,saturation_value):
	flat_rgb_image = np.array(rgb_image.reshape(1,width*height,3)[0], dtype="float")
	flat_rgb_image = np.divide(flat_rgb_image, 255)
	
	hsv_image = matplotlib.colors.rgb_to_hsv(flat_rgb_image)
	hsv_image[:,1]+=saturation_value
	hsv_image[hsv_image > 1] -= 1
	hsv_image[hsv_image < 0] = 0

	temp_rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)
	new_rgb_image = np.multiply(temp_rgb_image, 255)
	
	new_rgb_image = new_rgb_image.reshape(width, height, 3).astype(int)
	return new_rgb_image

def getLookupTable(x, y):
	spline = UnivariateSpline(x, y)
	return spline(range(256))


def coldImage(image):
	upLookup = getLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
	downLookup = getLookupTable([0, 64, 128, 256], [0, 50, 100, 256])

	red_channel, green_channel, blue_channel = cv2.split(image)
	red_channel = cv2.LUT(red_channel, downLookup).astype(np.uint8)
	blue_channel = cv2.LUT(blue_channel, upLookup).astype(np.uint8)
	
	return cv2.merge((red_channel, green_channel, blue_channel))


def warmImage(image):
	upLookup = getLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
	downLookup = getLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
	
	red_channel, green_channel, blue_channel = cv2.split(image)
	red_channel = cv2.LUT(red_channel, upLookup).astype(np.uint8)
	blue_channel = cv2.LUT(blue_channel, downLookup).astype(np.uint8)

	return cv2.merge((red_channel, green_channel, blue_channel))