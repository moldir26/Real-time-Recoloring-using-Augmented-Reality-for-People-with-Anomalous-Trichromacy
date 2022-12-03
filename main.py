#   ROBT310:    Project – Final Report
#   Name:       Dias Manap,     Moldir Berkaliyeva,     Zhanel Yessirkepbaeva
#   Student ID: 201871578,      201730450,              201824656

import cv2
import numpy as np
import imutils
import filter
from enum import Enum

# Video capture commands
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if video is showing
if (not capture.isOpened()):
	print ("Error: Can't find Camera")
	quit()
else:
	print ("Success: Camera is open")

# Types of colorblindness
class Type(Enum):
    PROTANOMALY = 1
    PROTANOPIA = 2
    DEUTERANOMALY = 3
    DEUTERANOPIA = 4
    TRITANOMALY = 5
    TRITANOPIA = 6
    ACHROMATOMALY = 7
    MONOCHROMACY = 8

# States of the app
class State(Enum):
    ORIGINAL = 1			    # original video
    SIMULATED = 2		        # simulated video for each type
    CORRECTED = 3		        # corrected video
    CONTOUR = 4		            # contour of RGBY
    RECOGNITION = 5             # center color recognition

# variables
state = State.CONTOUR         # current state
current_type = Type.PROTANOMALY  # current type of color blindness selected

def filter_image():
    _ , image = capture.read()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if state == State.ORIGINAL:
        new_rgb_image = image
    elif state == State.SIMULATED:
        new_rgb_image = filter.simulate(rgb_image, current_type)
    elif state == State.CORRECTED:
        if current_type == Type.PROTANOPIA:
            new_rgb_image = filter.correct_opia(rgb_image, .1)
        elif current_type == Type.PROTANOMALY:
            new_rgb_image = filter.coldImage(rgb_image)
        elif current_type == Type.DEUTERANOPIA:
            new_rgb_image = filter.correct_opia(rgb_image, .1)
        elif current_type == Type.DEUTERANOMALY:
            new_rgb_image = filter.coldImage(rgb_image)
        elif current_type == Type.TRITANOPIA:
            new_rgb_image = filter.correct_opia(rgb_image, .3)
        elif current_type == Type.TRITANOMALY:
            new_rgb_image = filter.correct_opia(rgb_image, .03)
        elif current_type == Type.ACHROMATOMALY:
            new_rgb_image = filter.warmImage(rgb_image)
        else:
            new_rgb_image = rgb_image
    elif state == State.CONTOUR:
        new_rgb_image = contour_rgby()
    elif state == State.RECOGNITION:
        new_rgb_image = recognize_middle()

    if state in [State.SIMULATED, State.CORRECTED]:
        new_rgb_image = new_rgb_image.astype(np.uint8)
        new_rgb_image = cv2.cvtColor(new_rgb_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("filter", new_rgb_image)

def contour_rgby():
    _ , image= capture.read()
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_lower = np.array([170,140,160])
    red_upper = np.array([180,255,255])

    green_lower = np.array([40,70,80])
    green_upper = np.array([70,255,255])

    blue_lower = np.array([90,60,0])
    blue_upper = np.array([121,255,255])

    yellow_lower = np.array([25,70,120])
    yellow_upper = np.array([30,255,255])

    
    mask_red = cv2.inRange(hsv,red_lower,red_upper)
    mask_green = cv2.inRange(hsv,green_lower,green_upper)
    mask_blue = cv2.inRange(hsv,blue_lower,blue_upper)
    mask_yellow = cv2.inRange(hsv,yellow_lower,yellow_upper)

    contours = []
    for mask_color in [(mask_red, "RED"), (mask_green, "GREEN"), (mask_blue, "BLUE"), (mask_yellow, "YELLOW")]:
        (mask, color_name) = mask_color
            
        contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.append((imutils.grab_contours(contour), color_name))
    
    for contour_color in contours:
        (contour, color) = contour_color
        for c in contour:
            area = cv2.contourArea(c)
            if area > 4500:
                highlight_color = (0,0,0)
                if color == "RED":
                    highlight_color = (0,0,255)
                elif color == "GREEN":
                    highlight_color = (0,255,0)
                elif color == "BLUE":
                    highlight_color = (255,0,0)
                elif color == "YELLOW":
                    highlight_color = (0,255,255)
                cv2.drawContours(image,[c],-1,highlight_color, 3)

                M = cv2.moments(c)

                center_x = int(M["m10"]/ M["m00"])
                center_y = int(M["m01"]/ M["m00"])

                cv2.circle(image,(center_x,center_y),7,(255,255,255),-1)
                cv2.putText(image, color, (center_x-20, center_y-20), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255,255,255), 3)

    return image

def recognize_middle():
    _ , image= capture.read()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, _ = image.shape
    center_x = int(width / 2)
    center_y = int(height / 2)
    
    center_hsv = hsv[center_y, center_x]
    hue = center_hsv[0]
    intensity = center_hsv[2]
    
    color_text = ""
    if intensity > 235: 
        color_text = "WHITE"
    elif intensity < 20:
        color_text = "BLACK"
    else:
        if hue < 5:
            color_text = "RED"
        elif hue < 33:
            color_text = "YELLOW"
        elif hue < 78:
            color_text = "GREEN"
        elif hue < 170:
            color_text = "BLUE"
        else:
            color_text = "RED"
    center_bgr = image[center_y, center_x]
    b, g, r = int(center_bgr[0]), int(center_bgr[1]), int(center_bgr[2])

    cv2.rectangle(image, (center_x - 220, 10), (center_x + 200, 120), (255, 255, 255), -1)
    cv2.putText(image, color_text, (center_x - 200, 100), 0, 3, (b, g, r), 5)
    cv2.circle(image, (center_x, center_y), 5, (25, 25, 25), 3)
    return image

while True:
    filter_image()

    k = cv2.waitKey(5)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()