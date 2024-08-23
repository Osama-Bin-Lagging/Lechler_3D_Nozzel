#### Set System Mode
PRODUCTION_MODE = True
DEBUG_MODE = False

#### IMPORT LIBRARIES
import cv2
import numpy as np
import liqdist_archit as ld
from liqdist_archit import detect_arucos
ld.DEBUG_MODE      = DEBUG_MODE
ld.PRODUCTION_MODE = PRODUCTION_MODE

if DEBUG_MODE:
    import matplotlib.pyplot as plt
    print("You are in debugging mode.")
    print("Multiple input streams are not supported")
    print("There will be lots of intermediate steps being printed out")

capture, selector = ld.select_ipcamera()
camera_matrix, distortion_coefficients = ld.read_cam_calibration()

if DEBUG_MODE:
    print("Image captured")
    plt.figure(figsize=(18,18))
    plt.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
# # JUPYTER: The following line is used as a method to track the locations of the aruco markers in live screen
# capture = ld.track(capture,camera_matrix,distortion_coefficients)
img_intrinsic = ld.intrinsic(capture,camera_matrix,distortion_coefficients)
# if DEBUG_MODE:
#     print("Image after aruco tracked")
#     plt.figure(figsize=(18,18))
#     plt.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

img_intrinsic = capture
if DEBUG_MODE:
    print("Image after Undistortion")
    plt.figure(figsize=(18,18))
    plt.imshow(cv2.cvtColor(img_intrinsic, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
arucoFound = ld.detect_arucos(capture,camera_matrix,distortion_coefficients)
if DEBUG_MODE:
    if arucoFound is not None:
        print("No of Aruco found: ",len(arucoFound))
    print("Normal image expects 4 aruco detections and live camera for some reason needs 8")
    print("The detected arucos are: ",arucoFound)

img_cr = ld.crop_image(img_intrinsic,arucoFound)
if DEBUG_MODE:
    print("Cropped Images")
    plt.figure(figsize=(18,18))
    plt.imshow(cv2.cvtColor(img_cr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()  
img_raw = ld.morphologic(img_cr)
if DEBUG_MODE:
    print("Image Morphed")
    plt.figure(figsize=(18,18))
    plt.imshow(img_raw)
    plt.axis('off')
    plt.show() 
# print(cv2.threshold(img_raw,127,255,cv2.THRESH_BINARY))
balls_found = ld.find_balls(img_raw,img_cr, 'balls','data',1)

