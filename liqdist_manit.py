# Import the necessary libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
from cv2 import aruco
import yaml
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

PRODUCTION_MODE = False
DEBUG_MODE = True

# Example usage
filename = "10.130.191.134_01_20240628141844680_1.mp4"
#vid = cv2.VideoCapture(f'./Vids/{filename}')

def detect_aruco_closest_frame(vid, output_dir="output_frames", cooldown_time=5, reloading_time = 10):
    max_frame = None
    frame_counter = 0
    frame_id = 1 
    cooldown_frames = int(30 * cooldown_time)  # Assuming 30 FPS
    reloading_frames = 0

    # Load ArUco dictionary and parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    aruco_params = aruco.DetectorParameters()

    previous_area = 0
    previous_previous_area = 0

    timestamps = []
    areas = []
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #arucodetector = aruco.ArucoDetector()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) == 4:
            current_area = compute_marker_area(corners)

            # Store timestamp and area for plotting
            timestamp = vid.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp / 1000)  # Convert to seconds
            areas.append(current_area)

            # Check for local maximum
            if previous_area > previous_previous_area and previous_area > current_area and frame_counter >= cooldown_frames and previous_area > 260000  and reloading_frames == 0:
                max_frame = frame.copy()
                
                print(f"Local max area found at {timestamp / 1000:.1f}s with area {previous_area:.1f} pixels")
                save_frame(max_frame, output_dir, filename, previous_area, frame_id)
                reloading_frames = int(30 * reloading_time)
                frame_id += 1  # Increment the frame ID for the next save
                frame_counter = 0  # Reset cooldown
            
            reloading_frames -= 1
            reloading_frames = max(0, reloading_frames)
            # Update areas
            previous_previous_area = previous_area
            previous_area = current_area

        frame_counter += 1  # Increment frame counter

    vid.release()
    cv2.destroyAllWindows()

    # Plot the areas vs timestamps
    #plot_areas_vs_time(timestamps, areas, filename)
    
    return

def compute_marker_area(corners):
    centers = [np.mean(corner, axis=1)[0] for corner in corners]
    centroid = np.mean(centers, axis=0)
    angles = [np.arctan2(center[1] - centroid[1], center[0] - centroid[0]) for center in centers]
    centers = [center for _, center in sorted(zip(angles, centers))]
    centers.append(centers[0])  # Close the loop
    return 0.5 * abs(sum(x1 * y2 - x2 * y1 for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])))

def save_frame(frame, output_dir, video_name, area, frame_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.splitext(video_name)[0]
    cv2.imwrite(f'{output_dir}/{frame_id}_max_area_frame.png', frame)
    

def plot_areas_vs_time(timestamps, areas, filename):
    # Convert lists to a pandas DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'Area': areas
    })
    
    # Export to CSV
    data.to_csv(f'areas_vs_time_{filename}.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, areas, marker='o', linestyle='-', color='b')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Area (pixels)')
    plt.title('Area of ArUco Markers vs. Time')
    plt.grid(True)
    plt.savefig(f'areas_vs_time_{filename}.png')
    plt.show()


#Additionally required: camera calibration file
################################################### ##########################
#Read camera calibration file

def read_cam_calibration(): 
    with open("Callibs/callibration.yaml", "r") as f:
        read_data = yaml.load(f, Loader=yaml.FullLoader)
        camera_matrix = np.array(read_data['camera_matrix'])
        distortion_coefficients = np.array(read_data['dist_coeff'])
        if DEBUG_MODE:
            print('read succesfully')
        return(camera_matrix, distortion_coefficients)

################################################### ##########################
#Selecting the IP camera (cameras should be labeled)
#Untested from my side 
def select_ipcamera(selector_init):
    selector = 0
    selected_cap = None
    if selector_init[0] == '/':
        selector = 5
        file_path = selector_init
        print("Image Path:", file_path)  # Debugging print
        selected_cap = cv2.VideoCapture(file_path)
    else:
        selector = int(selector_init[1])
        if selector == '1': 
            selected_cap = cv2.VideoCapture('rtsp://admin:lechler@123@10.130.191.134/Streaming/channels/2')
        elif selector == '2': 
            selected_cap = cv2.VideoCapture('rtsp://admin:L3chl3rGmbH@10.49.235.169:80')
        elif selector == '3': 
            selected_cap = cv2.VideoCapture('rtsp://admin:L3chl3rGmbH@10.49.235.171:80')
        elif selector == '4':
            selected_cap = cv2.VideoCapture('rtsp://admin:LechlerREA@10.49.235.46:80')
        elif selector == '6':
            selected_cap = cv2.VideoCapture(0)
            if selected_cap is None:
                print('Error loading image')
            else:
                print('Image loaded successfully')  # Debugging print
                pass
        else:
            print('Invalid Selection')
            selected_cap = 0
    return selected_cap, str(selector)


def detect_arucos(image, camera_matrix, distortion_coefficients):
    gray = image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Create a parameters dictionary and set values manually
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minMarkerDistanceRate = 0.05
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMethod = 0
    parameters.markerBorderBits = 1
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    parameters.maxErroneousBitsInBorderRate = 0.04
    parameters.minOtsuStdDev = 5.0
    parameters.errorCorrectionRate = 0.6

    detector =  cv2.aruco.ArucoDetector(aruco_dict, parameters)
    #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if DEBUG_MODE:    
        print("Detected ArUco Markers:")
        print("IDs:", ids)
        print("Corners:", corners)
    
    if np.all(ids is not None):
        if len(ids) != 4:
            print('ArUcoMarker error. Find',len(ids),' of four.')
            return None
        elif len(ids) ==4:
            if DEBUG_MODE:
                print('4 ArUco marker detected.')
            #Sort markers by identities in ascending order ---> otherwise subsequent assignment RL difficult +
            zipped = zip(ids, corners)
            ids, corners = zip(*(sorted(zipped)))
            corners = np.asarray(corners)
            ids = np.asarray(ids)
            # frame_markers = aruco.drawDetectedMarkers(gray.copy(), corners, ids)
            marker_centers = []
            for i in range(len(ids)):
                c = corners[i][0]
                d = [(c[0][0] + c[2][0]) / 2, (c[0][1] + c[2][1]) / 2]
                marker_centers.append(d)
            return marker_centers
    else: 
        print('No Aruico Marker Found please check camera position.')
        return None #return none instead of zero
##################################
#Perspective Transformation #
def crop_image(img, arucofound):
    marker_centers = np.array(arucofound)
    print(marker_centers)
    marker_centers[0:2] -= [0,40]
    marker_centers[2:4] += [40,0]

    # define size of new image in pixels
    rows = 2500
    cols = 5000
    
    # Source Points - coordinates of detected ArUcos 
    src_points = np.float32([
    marker_centers[0],  # Point 1
    marker_centers[1],  # Point 2
    marker_centers[2],  # Point 3
    marker_centers[3]   # Point 4
])
    #src_points += 20
    # Destination Points - destination points for the transformation (= measured real coordinates in mm in the local system) +
    dst_1 = [1000,1000]
    dst_2 = [4695,1000]
    dst_3 = [1000,1445]
    dst_4 = [4695,1445]
    dst_points = np.float32([dst_1, dst_2, dst_3, dst_4]) #build... ascending order since Arucos are sorted!
    
    # Determine the transformation matrix with SourcePoints and DestinationPoints
    affine_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    #Perform perspective transformation +
     # cv.WarpPerspective(src, dst, mapMatrix, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray =img
    img_warp = cv2.warpPerspective(gray, affine_matrix, (cols,rows))
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Specify height and width for crop: Centers of the ArUco markers +
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    h = dst_3[1]-dst_1[1]
    w = dst_4[0]-dst_3[0]

    #Crop the image area so that only the measuring section can be seen.
    # img_crop = img_warp[dst_1[1]+200:dst_1[1]+h+110, dst_1[0]+260:dst_1[0]+w-400]
    img_crop = img_warp[dst_1[1] + int(h*0.01):dst_1[1]+h -int(h*0.01), dst_1[0] + int(w*0.03):dst_1[0]+w - int(w*0.03)]
    if DEBUG_MODE:
        print("Cropped image size",img_crop.shape)

    return (img_crop)

def update_frame(cap, lbl_video):
    ret, frame = cap.read()
    if ret:
        # Convert the frame to an image compatible with Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        # Update the label with the new frame
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=lbl_video.imgtk)

        # Store the current frame
        lbl_video.current_frame = frame

    # Schedule the next frame update
    lbl_video.after(1, lambda: update_frame(cap, lbl_video))

def process_image(img_cr):
    hsv = cv2.cvtColor(img_cr, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image

    kernel = np.ones((5, 5), np.uint8)

    # Apply erosion to remove small noise
    mask = cv2.erode(mask, kernel, iterations=2)

    # Apply dilation to restore the size of the object
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply closing to close small holes inside the foreground objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply opening to remove small objects from the foreground
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply Gaussian blur to the mask to further reduce noise
    #blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    res = cv2.bitwise_and(img_cr, img_cr, mask=mask)
    img_bw = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    each_cylinder_width_pix = img_bw.shape[1]//75
    return img_bw, cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1.1, minDist=each_cylinder_width_pix*3//4, param1=5, param2=6.5, minRadius = 3, maxRadius = 15) 
