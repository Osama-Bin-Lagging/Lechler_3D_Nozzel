
import cv2 
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import csv
from scipy.interpolate import UnivariateSpline
from cv2 import aruco

import plotly.graph_objects as go
from scipy.interpolate import RectBivariateSpline, griddata
import tkinter as tk
from tkinter import Button
from tkinter import ttk, messagebox, filedialog
from tkvideo import tkvideo
from PIL import Image, ImageTk
 
import liqdist_manit as ld
import liqdist_archit as lda

DEBUG_MODE = False
PRODUCTION_MODE = not DEBUG_MODE

ld.DEBUG_MODE      = DEBUG_MODE
ld.PRODUCTION_MODE = PRODUCTION_MODE

GUPTA_IS_GREAT = False

home_area = 260000

cooldown_time = 30
reloading_time = 5

fps = 30
height = [13,9]
frame_counter = 0
frame_id = 0
cooldown_frames = int(fps * cooldown_time)  
reloading_frames = int(fps * reloading_time)

previous_area = 0
previous_previous_area = 0
debug_image_counter = 0

# Load ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
aruco_params = aruco.DetectorParameters()

camera_matrix, distortion_coefficients = ld.read_cam_calibration()

dataf = pd.DataFrame(columns = np.arange(71))
output_dir = 'Cropped_img'

def production_mode_processing(frame, timestamp):
        global frame_counter, frame_id, reloading_frames, previous_area, previous_previous_area, dataf, ratio, height, debug_image_counter
        max_frame = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #arucodetector = aruco.ArucoDetector()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) == 4:
                current_area = ld.compute_marker_area(corners)
                # Store timestamp and area for plotting
                    # Check for local maximum
                #print(frame_counter, cooldown_frames, reloading_frames)
                if previous_area > previous_previous_area and previous_area > current_area and frame_counter >= cooldown_frames and previous_area > home_area * 0.95 and reloading_frames == 0:
                    max_frame = frame.copy()
                    print(f"Local max area found at {timestamp / 1000:.1f}s with area {previous_area:.1f} pixels")
                    frame_id += 1  # Increment the frame ID for the next save
                    frame_counter = 0  # Reset cooldown
                    reloading_frames = int(fps * reloading_time)
                    

                    # Update areas
                previous_previous_area = previous_area
                previous_area = current_area
                reloading_frames -= 1
                reloading_frames = max(0, reloading_frames)
        frame_counter += 1  # Increment frame counter

        if max_frame is not None:    
            if GUPTA_IS_GREAT is False:
                arucoFound = ld.detect_arucos(max_frame,camera_matrix,distortion_coefficients)
                img_cr = ld.crop_image(max_frame,arucoFound)
                if DEBUG_MODE or PRODUCTION_MODE:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    debug_image_counter += 1
                    cv2.imwrite('./'+ output_dir+'/detected_green_'+ str(debug_image_counter) + '.jpg', img_cr)
                #------------------------------------------------------------------------------------------
                
                img_bw, circles = ld.process_image(img_cr)
                # Find contours in the mask
                #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    circle_coordinates = []
                    for i in circles[0, :]:
                        center = [i[0],img_cr.shape[1] - i[1]]
                        circle_coordinates.append(center)
                        # Draw the outer circle
                        cv2.circle(img_bw, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # Draw the center of the circle
                        cv2.circle(img_bw, (i[0], i[1]), 2, (0, 0, 255), 3)
                circle_coordinates = np.asarray(circle_coordinates)
            
            elif GUPTA_IS_GREAT is True:  
                debug_image_counter += 1
                #img_intrinsic = lda.intrinsic(max_frame,camera_matrix,distortion_coefficients)
                arucoFound = lda.detect_arucos(max_frame,camera_matrix,distortion_coefficients)
                #plt.imshow(img_intrinsic)
                #plt.show()
                img_cr = lda.crop_image(frame,arucoFound)
                cv2.imwrite('./Cropped_img/detected_green_'+ str(debug_image_counter)+'.jpg', img_cr)
                
                img_raw = lda.morphologic(img_cr)
                x_pos, y_height = lda.find_balls(img_raw,img_cr, 'balls', '1st', 1)
                circle_coordinates = np.stack((x_pos, y_height), axis=1)

            print(circle_coordinates)
            #circle_coordinates.sort(key=lambda x: x[0])
            print(np.argsort(circle_coordinates[:, 0]))
            circle_coordinates = circle_coordinates[np.argsort(circle_coordinates[:, 0])]
            circle_coordinates = np.array(circle_coordinates)
            x, y = circle_coordinates[2:-2,0], np.array(circle_coordinates[2:-2,1])
            #9
            pixel_height = [max(circle_coordinates[0:2,1]) - min(circle_coordinates[0:,1]), max(circle_coordinates[-2:,1]) - min(circle_coordinates[0:,1])]
            spline = UnivariateSpline(x, y, s= 3000)

            new_x = np.linspace(x[0], x[-1], 71)
            new_y = spline(new_x)
            ratio = (height[0] / pixel_height[0] + height[1] / pixel_height[0]) / 2
            dataf.loc[frame_id] = new_y
             
            if DEBUG_MODE or PRODUCTION_MODE:
                with open('circle_coordinates.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(new_y)


            # Plotting the original image with detected green points and spline
                plt.figure(figsize=(8, 6))
            #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.scatter(x, y, color='green', label='Detected Points', s=50)  # Adjust size here (s=50)
                plt.scatter(new_x, new_y, color='orange', label='Saved Points', s=50)  # Adjust size here (s=50)
                plt.plot(x, spline(x), color='red', label='Univariate Spline')
                plt.title('Univariate Spline on Green Points')
                plt.legend()
                plt.axis('off')  # Turn off axis for better visualization
                #plt.show()
                if not os.path.exists('final_splines'):
                    os.makedirs('final_splines')
                plt.savefig('./final_splines/detected_spline_'+ str(debug_image_counter) + '.jpg')

                if not os.path.exists('spline_with_img'):
                    os.makedirs('spline_with_img')
                plt.figure(figsize=(8, 6)) 
                plt.scatter(x, y, color='green', label='Detected Points', s=50)  # Adjust size here (s=50)
                plt.plot(x, spline(x), color='red', label='Univariate Spline')
                plt.imshow(cv2.cvtColor(img_cr, cv2.COLOR_BGR2RGB))
                plt.title('Univariate Spline on Image')
                plt.legend()
                plt.savefig('./spline_with_img/image_and_spline_'+ str(debug_image_counter) + '.jpg')

def plot_data_from_dataframe(dataf):
    global ratio
    row_min = dataf.min(axis=1)
    row_max = dataf.max(axis=1)
    threshold = row_max.max() * 0.2

    dataf = dataf.sub(row_min, axis=0)
    data = dataf.sort_index()
    #data.iloc[9][:69] -= 120
    mask = np.where(data <= threshold)[0]
    data[mask] = 0
    # Create a grid of coordinates for the smooth plot
    #y_smooth = np.linspace(0, y[-1], 100)
    #x_smooth, y_smooth = np.meshgrid(x, y_smooth)
    #z_smooth = spl(y_smooth)

    z = data.values * ratio
    y = data.columns.values * 2
    x = np.int32(data.index.values) * 15.6
    print(x,y)
    spline = RectBivariateSpline(x, y, z, s = 5000)
    z_smooth = pd.DataFrame(spline(x, y) ,index = np.int32(data.index.values))
    #print(z_smooth)
    #print(x.shape,y.shape,z.shape)
    #new_x = np.linspace(x[0], x[-1], 100)
    #new_y = np.linspace(y[0], y[-1], 100)
    #print(new_x,new_y)
    #z_smooth = spline(new_x, new_y)
    print(z_smooth)
    z_smooth.loc[0] = np.zeros_like(z_smooth.loc[1])
    z_smooth.loc[z_smooth.shape[0]] = np.zeros_like(z_smooth.loc[1])
    z_smooth = z_smooth.sort_index()

    x = np.int32(z_smooth.index.values) * 15.6
    print(z_smooth)
    # Create the 3D surface plot

    fig = go.Figure(data=[go.Surface(z=z_smooth, x=y, y=x)])
    

   
    #grid_x, grid_y = np.mgrid[np.min(x):np.max(x):500j, np.min(y):np.max(y):500j]

    # Interpolate the data using griddata
    #Z_interp = griddata((x, y), z_smooth, (grid_x, grid_y), method='cubic')

    #z_threshold = 6

    # Create a mask for the z values that are above the threshold
    #mask = Z_interp.ravel() > z_threshold

    # Create a DataFrame from the interpolated data, applying the mask
    #df_interp = pd.DataFrame({
    #    'x': grid_x.ravel()[mask],
    #    'y': grid_y.ravel()[mask],
    #    'z': Z_interp.ravel()[mask]
    #})

    # Create a pivot table from the DataFrame
    #pivot_table = df_interp.pivot(index='y', columns='x', values='z')

    # Create a heatmap from the pivot table
    #heatmap_fig = go.Figure(data=go.Heatmap(
    #                z=pivot_table.values,
    #                x=pivot_table.columns,
    #                y=pivot_table.index,
    #                colorscale='Viridis'))

    # Create the heatmap
    heatmap_fig = go.Figure(data=[go.Heatmap(x = y, y = x, z = z_smooth, colorscale='Viridis')])

    # Update the layout for the heatmap
    heatmap_fig.update_layout(title='Heatmap of Z Values',
                            xaxis_title='Y Axis',
                            yaxis_title='X Axis')

    heatmap_fig.show()

    fig.update_layout(scene=dict(
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        zaxis_title='Z Axis'),
                        title='3D Scatter Plot')
    
    fig.show()
    #fig.show()
    if not os.path.exists("images"):
        os.mkdir("images")
    
    fig.write_image("images/3Dplot.pdf")
    heatmap_fig.write_image('images/heatmap.pdf')

def open_config_window():
    config_window = tk.Toplevel(root)
    config_window.title("CONFIG")
    config_window.geometry("400x600")

    # Create a LabelFrame for mode selection
    mode_frame = ttk.LabelFrame(config_window, text="Select Mode")
    mode_frame.pack(padx=10, pady=10, fill="x", expand="yes")

    # Mode selection
    mode_var = tk.StringVar(value="PRODUCTION_MODE")
    mode_label = ttk.Label(mode_frame, text="Mode:")
    mode_label.pack(anchor="w", padx=5, pady=5)
    mode_menu = ttk.Combobox(mode_frame, textvariable=mode_var, values=["PRODUCTION_MODE", "DEBUG_MODE"])
    mode_menu.pack(anchor="w", padx=5, pady=5)

    # Create a LabelFrame for number inputs
    input_frame = ttk.LabelFrame(config_window, text="Enter Pixel to Height (mm) Ratios for Testtubes: ")
    input_frame.pack(padx=10, pady=10, fill="x", expand="yes")

    # Number 1 input
    num1_label = ttk.Label(input_frame, text="Height of Testtube 2: ")
    num1_label.pack(anchor="w", padx=5, pady=2)
    num1_entry = ttk.Entry(input_frame)
    num1_entry.pack(anchor="w", padx=5, pady=2)

    # Number 2 input
    num2_label = ttk.Label(input_frame, text="Height of Testtube 74: ")
    num2_label.pack(anchor="w", padx=5, pady=2)
    num2_entry = ttk.Entry(input_frame)
    num2_entry.pack(anchor="w", padx=5, pady=2)

    sleep1_label = ttk.Label(input_frame, text="Sleep time after reaching home (default is 5s): ")
    sleep1_label.pack(anchor="w", padx=5, pady=2)
    sleep1_entry = ttk.Entry(input_frame)
    sleep1_entry.pack(anchor="w", padx=5, pady=2)

    sleep2_label = ttk.Label(input_frame, text="Sleep time after each detection (default is 30s): ")
    sleep2_label.pack(anchor="w", padx=5, pady=2)
    sleep2_entry = ttk.Entry(input_frame)
    sleep2_entry.pack(anchor="w", padx=5, pady=2)

    def on_submit():
        global height, DEBUG_MODE, PRODUCTION_MODE, cooldown_time, reloading_time
        mode = mode_var.get()
        if num1_entry.get() is not None:
            num1 = num1_entry.get()
        if num2_entry.get() is not None:
            num2 = num2_entry.get()
        if sleep1_entry is not None:
            cooldown_time = sleep1_entry
        if sleep2_entry is not None:
            reloading_time = sleep2_entry

        try:
            num1 = float(num1)
            num2 = float(num2)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers.")
            return

        global inputs
        inputs['mode'] = mode
        inputs['num1'] = num1
        inputs['num2'] = num2

        height = [float(num1), float(num2)]
        

        if inputs["mode"] == 'DEBUG_MODE':
            DEBUG_MODE = True
        else:
            DEBUG_MODE = False

        PRODUCTION_MODE = not DEBUG_MODE

        ld.DEBUG_MODE      = DEBUG_MODE
        ld.PRODUCTION_MODE = PRODUCTION_MODE
        config_window.destroy()
        

    # Submit button
    submit_button = ttk.Button(config_window, text="Submit", command=on_submit)
    submit_button.pack(pady=10)

def start_video():
    options_frame.pack_forget()
    selector = inputs.get('option', None)
    if selector == "Webcam":
        start_cam()
    if selector[1] == '1': 
        start_cam('rtsp://admin:lechler@123@10.130.191.134/Streaming/channels/2')
    elif selector[1] == '2': 
        start_cam('rtsp://admin:L3chl3rGmbH@10.49.235.169:80')
    elif selector[1] == '3': 
        start_cam('rtsp://admin:L3chl3rGmbH@10.49.235.171:80')
    elif selector[1] == '4':
        start_cam('rtsp://admin:LechlerREA@10.49.235.46:80')
    elif selector[1] == '6':
        start_cam(0)
    elif selector[0] == '/':
        start_cam(selector)
    else:
        messagebox.showerror("Error", "No video file selected.")

def start_cam(arg):
    global fps, cooldown_frames, reloading_frames
    global cap, original_size
    cap = cv2.VideoCapture(arg)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    cooldown_frames = int(fps * cooldown_time)  # Assuming 30 FPS
    reloading_frames = int(fps * reloading_time)
    capture_button.pack(pady=10)
    show_frame()

def show_frame():
    global cap, is_paused
    if is_paused:
        return
    ret, frame = cap.read()
    if ret:
        new_frame = cv2.resize(frame, (video_label.winfo_width(), video_label.winfo_height()))
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(new_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if capture_frames:
            production_mode_processing(frame, cap.get(cv2.CAP_PROP_POS_MSEC))
        delay = int(1000 / fps)
        video_label.after(delay, show_frame)
    else:
        print('Video stream ended')
        cap.release()
        cv2.destroyAllWindows()

def capture_frame():
    global capture_frames, original_size, text_info, text_info_label, frame_counter
    frame_counter = 0
    capture_frames = True
    option = inputs.get('option', None)
    if option[0] != "/" and cap is not None:
        _, frame = cap.read()
    else:
        cap = cv2.VideoCapture(option)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        _, frame = cap.read()

    if frame is not None:
            if original_size is None:
                original_size = (frame.shape[1], frame.shape[0])
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
            #frame = cv2.resize(frame, (video_label.winfo_width(), video_label.winfo_height()))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #arucodetector = aruco.ArucoDetector()
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
            aruco_params = aruco.DetectorParameters()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
            if ids is not None and len(ids) == 4:
                global home_area
                home_area = ld.compute_marker_area(corners)
                text_info.set(f'Configured, home area = {home_area}')
                text_info_label.pack()
            else:
                print('Error brother')
            #cv2.destroyAllWindows()
            
            # Close the Tkinter window
    else:
        messagebox.showerror("Error", "Failed to capture frame.")

def end_video_options():
    end_video_dialog = tk.Toplevel(root)
    end_video_dialog.title("End Video")
    end_video_dialog.geometry("300x100")
    
    tk.Label(end_video_dialog, text="Choose an option:").pack(pady=5)
    
    plot_button = ttk.Button(end_video_dialog, text="PLOT", command=lambda: plot_function(end_video_dialog))
    plot_button.pack(side="left", padx=10, pady=10)
    
    discard_button = ttk.Button(end_video_dialog, text="DISCARD", command=lambda: reset_all(end_video_dialog))
    discard_button.pack(side="right", padx=10, pady=10)

def plot_function(dialog):
    dialog.destroy()
    # Call your plotting function here
    global dataf, is_paused
    print("Plotting function called.")
    plot_data_from_dataframe(dataf)
    is_paused = False

def end_video():
    global is_paused
    is_paused = True
    end_video_options()

def reset_all(end_video_dialog = None):
    if end_video_dialog is not None:
        end_video_dialog.destroy()
    global inputs, cap, original_size, capture_frames, fps, dataf, frame_counter, frame_id, cooldown_frames, reloading_frames, previous_area, previous_previous_area
    inputs = {}
    if cap:
        cap.release()
    cap = None
    original_size = None
    dataf = pd.DataFrame(columns = np.arange(71))
    capture_frames = False
    #fps = 30
    video_label.config(image='')
    frame_counter = 0
    frame_id = 0
    cooldown_frames = int(fps * cooldown_time)  # Assuming 30 FPS
    reloading_frames = int(fps * reloading_time)

    previous_area = 0
    previous_previous_area = 0
    print("All settings reset.")

def show_production_options():
        options_frame.pack_forget()
        options_frame.pack(padx=10, pady=10, fill="x", expand="yes")

        option_label = ttk.Label(options_frame, text="Choose an option:")
        option_label.pack(anchor="w", padx=5, pady=5)
        
        option_menu = ttk.Combobox(options_frame, values=["(1)...Camera IP 10.130.191.134", "(2)...Camera IP 10.49.235.169", "(3)...Camera IP 10.49.235.171", '(4)...Camera IP 10.49.235.46', "(5)...Input file from device", '(6)...Webcam'])
        option_menu.pack(anchor="w", padx=5, pady=5)
        option_menu.bind("<<ComboboxSelected>>", lambda event: on_option_submit(option_menu.get()))

def show_debug_options():
        options_frame.pack_forget()
        options_frame.pack(padx=10, pady=10, fill="x", expand="yes")

        option_button = ttk.Button(options_frame, text="Select File", command=select_file)
        option_button.pack(pady=10)

def on_option_submit(option):
        if option == "(5)...Input file from device":
            select_file()
        else:
            chosen_option.set(option)
            inputs['option'] = option

def display_options_based_on_mode():
    for widget in options_frame.winfo_children():
        widget.destroy()

    if PRODUCTION_MODE:
        show_production_options()
    else:
        show_debug_options()

def select_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            chosen_option.set(file_path)
            inputs['option'] = file_path

# Main window
root = tk.Tk()
root.title("Lechler GUI")
root.geometry("600x600")

# Video display area
video_frame = ttk.Frame(root)
video_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=400)

# Text info frame on the top right
text_info_frame = ttk.Frame(root)
text_info_frame.pack(side="top", anchor='ne', padx=10)

text_info = tk.StringVar()
text_info.set("Not configured yet")
text_info_label = ttk.Label(text_info_frame, textvariable=text_info, justify="left")
text_info_label.pack()

video_label = ttk.Label(video_frame)
video_label.pack(expand=True)

# Bottom buttons frame
buttons_frame = ttk.Frame(root)
buttons_frame.pack(fill="x", side="bottom", padx=10, pady=10)

# Configure button
configure_button = ttk.Button(buttons_frame, text="CONFIG", command=open_config_window)
configure_button.pack(side="left", padx=5, pady=5)

# Start video button
start_button = ttk.Button(buttons_frame, text="START VIDEO", command=start_video)
start_button.pack(side="right", padx=5, pady=5)

# Capture frame button
capture_button = ttk.Button(buttons_frame, text="CONFIGURE", command=capture_frame)
capture_button.pack(side="left", padx=5, pady=5)

# End video button
end_button = ttk.Button(buttons_frame, text="END VIDEO", command=end_video_options)
end_button.pack(side="right", padx=5, pady=5)

options_frame = ttk.LabelFrame(video_frame, text="Additional Options")
options_frame.pack(padx=10, pady=10, fill="x", expand="yes")

# Button to display additional options
additional_options_button = ttk.Button(video_frame, text="Select Source", command=lambda: display_options_based_on_mode())
additional_options_button.pack()
# Variable to store the chosen option
chosen_option = tk.StringVar()

# Dictionary to store all inputs
inputs = {}
inputs['num1'] = height[0]
inputs['num2'] = height[1]
cap = None
current_frame = 0
capture_frames = False
original_size = None
is_paused = False
# Run the application
root.mainloop()

# Print the chosen option after the window is closed

'''
if PRODUCTION_MODE:
    vid, _ = ld.select_ipcamera(inputs["option"])


def process_frame(img):
    print(4)
    if img is not None:
        print(1)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #arucodetector = aruco.ArucoDetector()
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        aruco_params = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        print(2)
        if ids is not None and len(ids) == 4:
            global home_area
            home_area = ld.compute_marker_area(corners)
        else:
            print('Error brother')
        #cv2.destroyAllWindows()
        
        # Close the Tkinter window
        root.destroy()
'''

#while True:
#   success, img = cap.read()
#   cv2.imshow("Result", img)
#   cv2.waitKey(0)
#cap.release()
#cv2.destroyAllWindows()

def debug_mode_processing(vid):
    filename = inputs["option"]
    vid = cv2.VideoCapture(filename)
    for file in os.listdir('.'):
        if file == 'circle_coordinates':
            os.remove('./circle_coordinates.csv')
        elif file == 'output_frames':
            for img in os.listdir('./'+ file):
                os.remove('./output_frames/'+ img)
    ld.detect_aruco_closest_frame(vid, cooldown_time=20, reloading_time = 10)
    output_dir = 'Cropped_img'
    images = os.listdir('./output_frames')
    for image in images:
        print("You are in debugging mode.")
        print("Multiple input streams are not supported")
        print("There will be lots of intermediate steps being printed out") 
        
        print("Importing image as a way to debug")
        ##FOR_DEBUG: Specify the image path
        image_path = "./output_frames/"+image
        print("Image Path entered: ",image_path)
        capture = cv2.imread(image_path)

        if DEBUG_MODE:
            print("Image captured")
            #plt.figure(figsize=(18,18))
            #print(capture.shape)
            #plt.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
            #plt.axis('off')
            #plt.show()

        arucoFound = ld.detect_arucos(capture,camera_matrix,distortion_coefficients)

        if arucoFound is not None:
            print("No of Aruco found: ",len(arucoFound))
            print("Normal image expects 4 aruco detections and live camera for some reason needs 8")
            print("The detected arucos are: ",arucoFound)
        img_cr = ld.crop_image(capture,arucoFound)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite('./'+ output_dir+'/detected_green_'+ image[:-19] + '.jpg', img_cr)

        img_bw, circles = ld.process_image(img_cr)
        # Find contours in the mask
        #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle_coordinates = []
            for i in circles[0, :]:
                center = [i[0],img_cr.shape[1] - i[1]]
                circle_coordinates.append(center)
                # Draw the outer circle
                cv2.circle(img_bw, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(img_bw, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        circle_coordinates.sort(key=lambda x: x[0])
        circle_coordinates = np.array(circle_coordinates)
        x, y = circle_coordinates[2:-2,0], np.array(circle_coordinates[2:-2,1])

        spline = UnivariateSpline(x, y, s= 3000)

        new_x = np.linspace(x[0], x[-1], 71)
        new_y = spline(new_x)

        dataf.loc[int(image[:-19])] = new_y


        with open('circle_coordinates.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_y)
        
        #plt.figure(figsize=(8, 6))
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.scatter(x, y, color='green', label='Detected Points', s=50)  # Adjust size here (s=50)
        #plt.scatter(new_x, new_y, color='orange', label='Saved Points', s=50)  # Adjust size here (s=50)
        #plt.plot(x, spline(x), color='red', label='Univariate Spline')
        #plt.title('Univariate Spline on Green Points')
        #plt.legend()
        #plt.axis('off')  # Turn off axis for better visualization
        #plt.show()

#print(dataf)

#files = [f for f in os.listdir('./balls') if f.endswith('.txt')]
#print(files)

#data = []
#for file in files:
#	with open('./balls/'+ file) as f:
#		s = f.read()
#		data.append(s.split())

#data = pd.read_csv('circle_coordinates.csv', header=None)

#data.iloc[8][-20:] += 990 - data.iloc[8][-23:]
#data.iloc[8][:51] -= 2050
#data = [[float(item) for item in row] for row in data]
#normalized_list = [[item - min(row) for item in row] for row in data]

   