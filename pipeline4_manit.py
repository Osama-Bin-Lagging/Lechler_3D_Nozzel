
import cv2 
import os
import numpy as np
import pandas as pd
import time
import tkinter as tk
from tkinter import Button
from tkinter import ttk, messagebox, filedialog
from tkvideo import tkvideo
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

import liqdist_manit as ld

DEBUG_MODE = False
PRODUCTION_MODE = not DEBUG_MODE

ld.DEBUG_MODE      = DEBUG_MODE
ld.PRODUCTION_MODE = PRODUCTION_MODE
debug_image_counter = 0
counter = 0
ball_coordinates = pd.DataFrame(columns = np.arange(71))
frame = None
height = [13,9]
ratio = 1
imputs = {}
width_list = []
frame_id = 0

wait_time = 10

dir_name = os.getcwd()

def draw_black_box(c, center_x, center_y, aspect, max_width, max_height):
    # Calculate the box dimensions while maintaining aspect ratio
    if aspect >= 1:  # Wider than tall
        box_width = min(max_width, max_height / aspect)
        box_height = box_width * aspect
    else:  # Taller than wide
        box_height = min(max_height, max_width * aspect)
        box_width = box_height / aspect

    # Calculate the bottom-left coordinates to center the box
    x = center_x - box_width / 2
    y = center_y - box_height / 2

    # Set the fill color to black and draw the rectangle
    c.setFillColor(colors.black)
    c.setLineWidth(1)
    c.rect(x, y, box_width, box_height, fill=False)

def draw_user_input_table(c, data, x, y):
    # Create a Table object
    data = [[str(key), str(value)] for key, value in data.items()]
    table = Table(data, colWidths=[50 * mm, 50 * mm])

    # Style the table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    table.setStyle(style)

    # Calculate the table height
    table_height = len(data) * 12 * mm  # Approximate each row height
    table.wrapOn(c, x, y)  # Prepare table size calculations
    table.drawOn(c, x, y - table_height)  # Draw the table

def draw_image_with_aspect(c, image_path, center_x, center_y, max_width, max_height):
    # Read the image
    image = ImageReader(image_path)
    img_width, img_height = image.getSize()
    aspect = img_height / float(img_width)

    # Determine new width and height preserving the aspect ratio
    if img_width > img_height:
        draw_width = min(max_width, img_width * (max_height / img_height))
        draw_height = draw_width * aspect
    else:
        draw_height = min(max_height, img_height * (max_width / img_width))
        draw_width = draw_height / aspect

    # Calculate the bottom-left coordinates to center the image
    x = center_x - draw_width / 2
    y = center_y - draw_height / 2

    # Draw the image on the canvas
    c.drawImage(image_path, x, y, width=draw_width, height=draw_height)
    return draw_width, draw_height, center_x, center_y

def generate_pdf(user_inputs):
    file_path =  os.path.join(os.getcwd(), "output_document.pdf")
    c = canvas.Canvas(file_path, pagesize=landscape(A4))
    width, height = landscape(A4)

    # Add user inputs on the left-hand side
    draw_user_input_table(c, user_inputs, x=20 * mm, y=height + 35 * mm)

    bottom_image_width, bottom_image_height, bottom_center_x, bottom_center_y = draw_image_with_aspect(c, "./images/Heatmap.png", width / 2, height / 2 - 50 * mm, max_width=220 * mm, max_height=110 * mm)
    right_image_width, right_image_height, right_center_x, right_center_y = draw_image_with_aspect(c, "./images/3Dplot.png", width / 2 + 70 * mm, height / 2 + 40 * mm, max_width=150 * mm, max_height=120 * mm)
    
    draw_black_box(c, right_center_x, right_center_y, aspect=right_image_height / right_image_width / 1.4, max_width=right_image_width / 1, max_height=right_image_height / 1.2)
    draw_black_box(c, bottom_center_x, bottom_center_y - 3 * mm, aspect=bottom_image_height / bottom_image_width / 1.4, max_width=bottom_image_width / 1, max_height=bottom_image_height/ 1.1)

    draw_image_with_aspect(c, './logo.png', width - 55 * mm, height - 15 * mm, max_width=100 * mm, max_height=100 * mm)

    # Save the PDF
    c.save()
    #import shutil

    #source = "../output_document.pdf"
    #destination = "./output_document.pdf"

    # Move the file
    #shutil.move(source, destination)

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

    width_label = ttk.Label(input_frame, text="Width between readings: ")
    width_label.pack(anchor="w", padx=5, pady=2)
    width_entry = ttk.Entry(input_frame)
    width_entry.pack(anchor="w", padx=5, pady=2)


    def on_submit():
        global height, DEBUG_MODE, PRODUCTION_MODE
        mode = mode_var.get()
        if num1_entry.get() is not None:
            num1 = num1_entry.get()
        if num2_entry.get() is not None:
            num2 = num2_entry.get()
        if width_entry.get() is not None:
            width = width_entry.get()

        try:
            num1 = float(num1)
            num2 = float(num2)
            width = float(width)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers.")
            return

        global inputs
        inputs['mode'] = mode
        inputs['num1'] = num1
        inputs['num2'] = num2
        inputs['width'] = width

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
    video_label.config(image='')
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
    global frame, height, debug_image_counter, ratio, frame_id, counter
    debug_image_counter += 1
    frame_id += 1
    new_y, ratio, width = ld.image_to_new_y(frame, height, debug_image_counter)
    counter += 1
    width_list.append(width)
#---------------------------------------------------------------------------------------------------------------------------------
# NEED TO FIX WIDTH
#---------------------------------------------------------------------------------------------------------------------------------
    new_y = centered(new_y)
    ball_coordinates.loc[frame_id] = new_y

def centered(new_y):
    min_y = np.min(new_y)

    new_y -= min_y
    #cumulative_sum = np.cumsum(new_y)
    #half_sum = cumulative_sum[-1] / 2
    #center_x = np.searchsorted(cumulative_sum, half_sum)
    center_x = round(np.sum(new_y * np.arange(len(new_y))) / np.sum(new_y)) 
    print(center_x, min_y)
    new_y += min_y
    if center_x > 35:
        new_y[:71 - (center_x - 35)] = new_y[center_x - 35:]
        new_y[71 - (center_x - 35):] = min_y
    elif center_x < 35:
        new_y[35 - center_x:] = new_y[: 71 - (35 - center_x)]
        new_y[:35 - center_x] = min_y
    return new_y

def open_folder(selector):
    global frame_id, ratio, ball_coordinates, debug_image_counter, final_width, width_list
    list1 = os.listdir(selector)
    #list1 = [item for item in list1 if not item.startswith('.')]
    image_extensions = {'.jpg', '.jpeg', '.png'}
    list1 = [file for file in list1 if os.path.splitext(file)[1].lower() in image_extensions]
    list1.sort()
    print(list1) 
    counter = -1
    width_list = np.zeros(len(list1))
    for image in list1:
        frame_id += 1 
        image_path = os.path.join(selector, image)
        frame = cv2.imread(image_path)
        debug_image_counter += 1
        new_y, ratio, width = ld.image_to_new_y(frame, height, debug_image_counter)
        counter += 1
        print(max(new_y) - min(new_y))
        width_list[counter] = width
        new_y = centered(new_y)
        ball_coordinates.loc[frame_id] = new_y
    print(width_list)
    final_width = np.average(width_list)
        
def plot_function(dialog):
    global inputs, width_list
    dialog.destroy()
    # Call your plotting function here
    global ball_coordinates, ratio
    print("Plotting function called.")
    
    spray_width, spray_depth, spray_angle = ld.plot_3dplot_heatmap(ball_coordinates, ratio, inputs['width'], wait_time)
    video_label.config(image='')
    final_width = np.average(width_list[1:-1])
    if final_width>400:
        final_width = final_width * 0.9
    elif final_width<300:
        final_width = final_width * 1.2
    print('asdfghjkjhgfdsasdfghjkhgfds', np.average(width_list), np.average(width_list[1:-1]))
    reset_all()
    #final_width, final_depth = max(spray_width, spray_depth), min(spray_width, spray_depth)
    user_inputs = {
    "Pro. No.": "197.079.17.32.00.0",
    "Date": time.strftime("%d/%m/%Y"),
    "Water pressure": "4.00 bar",
    "Air pressure": "2.00 bar",
    "Spray height": "200 mm",
    "Water flow rate": "6.60 l/min",
    "Air flow rate": "7.15 m³/h",
    "Average value": "8.45 ml",
    'Spray Angle': f'{round(spray_angle, 3)} degrees',
    'Spray Width': f'{round(final_width, 3)} mm'
    #,'Spray depth': f'{round(spray_depth, 3)} mm'
    }
    generate_pdf(user_inputs)

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
    global inputs, cap, fps, ball_coordinates, frame_id, is_paused, debug_image_counter
    debug_image_counter = 0
    is_paused = True
    #inputs = {}
    #if cap:
    #    cap.release()
    #cap = None
    ball_coordinates = pd.DataFrame(columns = np.arange(71))
    fps = 30
    frame_id = 0
    display_logo()
    print("All settings reset.")
    
def display_logo():
    try:
        # Open the image
        logo_img = Image.open("logo.png")
        
        # Calculate the aspect ratio and resize
        target_width, target_height = 600, 400
        original_width, original_height = logo_img.size
        aspect_ratio = original_width / original_height

        # Determine the new dimensions maintaining the aspect ratio
        if (target_width / target_height) > aspect_ratio:
            # Fit to height
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            # Fit to width
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        # Resize the image with LANCZOS filter for high-quality downsampling
        logo_img = logo_img.resize((new_width, new_height), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_img)

        # Configure the label to display the image
        video_label.config(image=logo_photo)
        video_label.image = logo_photo  # Keep a reference to avoid garbage collection
    except Exception as e:
        print(f"Error loading logo: {e}")

root = tk.Tk()
root.title("Lechler GUI")
root.geometry("600x600")

# Video display area
video_frame = ttk.Frame(root)
video_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=400)

video_label = ttk.Label(video_frame)
video_label.pack(fill="both", expand=True)

display_logo()

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
inputs['width'] = 20
cap = None
current_frame = 0
is_paused = False
# Run the application
root.mainloop()