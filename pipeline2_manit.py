
import cv2 
import os
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import Button
from tkinter import ttk, messagebox, filedialog
from tkvideo import tkvideo
from PIL import Image, ImageTk

import liqdist_manit as ld

DEBUG_MODE = False
PRODUCTION_MODE = not DEBUG_MODE

ld.DEBUG_MODE      = DEBUG_MODE
ld.PRODUCTION_MODE = PRODUCTION_MODE
debug_image_counter = 0
ball_coordinates = pd.DataFrame(columns = np.arange(71))
frame = None
height = [13,9]
ratio = 1
imputs ={}

frame_id = 0

dir_name = os.getcwd()

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

    def on_submit():
        global height, DEBUG_MODE, PRODUCTION_MODE
        mode = mode_var.get()
        if num1_entry.get() is not None:
            num1 = num1_entry.get()
        if num2_entry.get() is not None:
            num2 = num2_entry.get()

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
    elif selector[0] == 'f':
         open_folder(selector[1:])
    else:
        messagebox.showerror("Error", "No video file selected.")

def start_cam(arg):
    global fps
    global cap
    cap = cv2.VideoCapture(arg)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    capture_button.pack(pady=10)
    show_frame()

def show_frame():
    global cap, is_paused, frame
    if is_paused:
        return
    if cap is None:
         return
    ret, frame = cap.read()
    if ret:
        new_frame = cv2.resize(frame, (video_label.winfo_width(), video_label.winfo_height()))
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(new_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        delay = int(1000 / fps)
        video_label.after(delay, show_frame)
    else:
        print('Video stream ended')
        cap.release()
        cv2.destroyAllWindows()

def file_option(end_video_dialog):
    global inputs
    end_video_dialog.destroy()
    file_path = filedialog.askopenfilename(initialdir = dir_name)
    if file_path:
            chosen_option.set(file_path)
            inputs['option'] = file_path

def folder_option(end_video_dialog):
    end_video_dialog.destroy()
    global inputs
    file_path = filedialog.askdirectory(initialdir = dir_name)
    if file_path:
            chosen_option.set(file_path)
            inputs['option'] = 'f'+file_path

def select_file():
        end_video_dialog = tk.Toplevel(root)
        end_video_dialog.title("Select video file or folder")
        end_video_dialog.geometry("300x100")
        
        tk.Label(end_video_dialog, text="Choose an option:").pack(pady=5)

        plot_button = ttk.Button(end_video_dialog, text="File", command=lambda: file_option(end_video_dialog))
        plot_button.pack(side="left", padx=10, pady=10)
        
        discard_button = ttk.Button(end_video_dialog, text="Folder", command=lambda: folder_option(end_video_dialog))
        discard_button.pack(side="right", padx=10, pady=10)

def show_production_options():
        options_frame.pack_forget()
        options_frame.pack(padx=10, pady=10, fill="x", expand="yes")

        option_label = ttk.Label(options_frame, text="Choose an option:")
        option_label.pack(anchor="w", padx=5, pady=5)
        
        option_menu = ttk.Combobox(options_frame, values=["(1)...Camera IP 10.130.191.134", "(2)...Camera IP 10.49.235.169", "(3)...Camera IP 10.49.235.171", '(4)...Camera IP 10.49.235.46', "(5)...Input file/folder from device", '(6)...Webcam', '(7)...Folder'])
        option_menu.pack(anchor="w", padx=5, pady=5)
        option_menu.bind("<<ComboboxSelected>>", lambda event: on_option_submit(option_menu.get()))

def show_debug_options():
        options_frame.pack_forget()
        options_frame.pack(padx=10, pady=10, fill="x", expand="yes")

        option_button = ttk.Button(options_frame, text="Select File", command=select_file)
        option_button.pack(pady=10)

def on_option_submit(option):
        if option == "(5)...Input file/folder from device":
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

def capture_frame():
    global frame, height, debug_image_counter, ratio
    debug_image_counter += 1
    new_y, ratio = ld.image_to_new_y(frame, height, debug_image_counter)

    ball_coordinates.loc[frame_id] = new_y

def open_folder(selector):
    global frame_id, ratio, ball_coordinates, debug_image_counter
    list1 = os.listdir(selector)
    list1.sort()
    print(list1) 
    for image in list1:
        frame_id += 1 
        image_path = os.path.join(selector, image)
        frame = cv2.imread(image_path)
        debug_image_counter += 1
        new_y, ratio = ld.image_to_new_y(frame, height, debug_image_counter)

        ball_coordinates.loc[frame_id] = new_y
        
def plot_function(dialog):
    dialog.destroy()
    # Call your plotting function here
    global ball_coordinates, ratio
    print("Plotting function called.")
    ld.plot_3dplot_heatmap(ball_coordinates, ratio)
    video_label.config(image='')

def end_video_options():
    end_video_dialog = tk.Toplevel(root)
    end_video_dialog.title("End Video")
    end_video_dialog.geometry("300x100")
    
    tk.Label(end_video_dialog, text="Choose an option:").pack(pady=5)
    
    plot_button = ttk.Button(end_video_dialog, text="PLOT", command=lambda: plot_function(end_video_dialog))
    plot_button.pack(side="left", padx=10, pady=10)
    
    discard_button = ttk.Button(end_video_dialog, text="DISCARD", command=lambda: reset_all(end_video_dialog))
    discard_button.pack(side="right", padx=10, pady=10)

def reset_all(end_video_dialog = None):
    if end_video_dialog is not None:
        end_video_dialog.destroy()
    video_label.pack()
    global inputs, cap, fps, ball_coordinates, frame_id, is_paused
    is_paused = True
    inputs = {}
    if cap:
        cap.release()
    cap = None
    ball_coordinates = pd.DataFrame(columns = np.arange(71))
    fps = 30
    frame_id = 0
    video_label.config(image='')
    print("All settings reset.")
    

root = tk.Tk()
root.title("Lechler GUI")
root.geometry("600x600")

# Video display area
video_frame = ttk.Frame(root)
video_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=400)

video_label = ttk.Label(video_frame)
video_label.pack(fill="both", expand=True)

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
capture_button = ttk.Button(buttons_frame, text="CAPTURE FRAME", command=capture_frame)
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
is_paused = False
# Run the application
root.mainloop()