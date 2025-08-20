import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import json

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# # Path to the dataset folder
image_dir = 'images_suplement'
dataset_folder = os.path.join(script_dir, image_dir)

# dataset_folder = '/home/pooh/igor_ws/src/my_own_thorvald/src/images'

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('img.png'))]
image_files.sort()  # Sort the files for consistent ordering
image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))

# Initialize global variables
start_x, start_y = None, None
image = None
mask = None
line_mask = None
pt1, pt2, val1, val2 = (0,0), (0,0), 0, 0
slider_vars = {}

# Function to keep the largest components in a binary mask
def keep_largest_components(binary_mask, num_components_to_keep):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    invert_mask = 255 - binary_mask

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a new binary mask with the X largest contours filled in
    result_mask = np.zeros_like(binary_mask)

    for i in range(min(num_components_to_keep, len(contours))):
        cv2.drawContours(result_mask, [contours[i]], -1, 255, thickness=cv2.FILLED)
    
    result_mask = np.where(invert_mask==255, 0 , result_mask)

    return result_mask

# Function to find the intersection points with the image borders
def find_edge_intersections(image_width, image_height, start, end):
    # Calculate the line equation coefficients: y = mx + b
    x1, y1 = start
    x2, y2 = end

    # Slope (m) of the line
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
    
    # Intercept (b) of the line
    b = y1 - m * x1 if m != np.inf else x1

    # List of intersections
    intersections = []

    # For vertical lines
    if m == np.inf:
        if 0 <= y1 <= image_height:
            intersections.append((int(np.round(x1)), 0))
            intersections.append((int(np.round(x1)), image_height))
            return intersections

    # Intersection with the left edge (x = 0)
    if m != np.inf:  # Avoid division by zero (vertical line)
        y_intersection = m * 0 + b
        if 0 <= y_intersection <= image_height:
            intersections.append((0, int(np.round(y_intersection))))

    # Intersection with the right edge (x = image_width)
    if m != np.inf:
        y_intersection = m * image_width + b
        if 0 <= y_intersection <= image_height:
            intersections.append((image_width, int(np.round(y_intersection))))
    
    # Intersection with the top edge (y = 0)
    if m != 0:  # Avoid horizontal line
        x_intersection = (0 - b) / m
        if 0 <= x_intersection <= image_width:
            intersections.append((int(np.round(x_intersection)), 0))

    # Intersection with the bottom edge (y = image_height)
    if m != 0:
        x_intersection = (image_height - b) / m
        if 0 <= x_intersection <= image_width:
            intersections.append((int(np.round(x_intersection)), image_height))

    return intersections

# Points to border value
def point_to_border(image_width, image_height, start, end):
    w = image_width
    h = image_height

    x1, y1 = start
    x2, y2 = end

    if y1>y2:
        # Swap points if the line is going downwards
        aux = x2, y2, x1, y1
        x1, y1, x2, y2 = aux
        #print("Swapping points to ensure y1 <= y2")
        #print(f"Start: ({x1},{y1}), End: ({x2},{y2})", f"Width: {w}, Height: {h} - before conversion - swapped")
    else:
        #print(f"Start: ({x1},{y1}), End: ({x2},{y2})", f"Width: {w}, Height: {h} - before conversion")
        pass
    
    if y2==h:
        if y1==0:
            #print("Line A-Alpha")
            val1 = h+x2
            val2 = h+x1
        elif x1==0:
            #print("Line B-Beta")
            val1 = h+x2
            val2 = h-y1
        elif x1==w:
            #print("Line C-Gama")
            val1 = h+x2
            val2 = h+w+y1
        else: 
            val1 = None
            val2 = None
    elif x2==0:
        if y1==0:
            #print("Line D-Alpha")
            val1 = y2
            val2 = h+x1
        elif x1==w:
            #print("Line E-Gama")
            val1 = y2
            val2 = h+w+y1
        else:
            val1 = None
            val2 = None
    elif x2==w:
        if y1==0:
            #print("Line F-Alpha")
            val1 = 2*h+w-y2
            val2 = h+x1
        elif x1==0:
            #print("Line G-Beta",h,y1,y2)
            val1 = 2*h+w-y2
            val2 = h-y1
        else:
            val1 = None
            val2 = None
    else:
        val1 = None
        val2 = None
    
    if val1 is not None and val2 is not None:
        #print(f"Intersection values: {val1}, {val2}")
        pass
    else:
        #print("No valid intersection values found.")
        val1 = 0
        val2 = 0

    return val1, val2

# Border values to points
def border_to_point(image_width, image_height, val1, val2):
    w = image_width
    h = image_height

    if val2<h:
        x1 = 0
        y1 = h-val2
    elif val2>h+w:
        x1 = w
        y1 = val2-h-w
    else:
        x1 = val2-h
        y1 = 0
    
    if val1<h:
        x2 = 0
        y2 = val1
    elif val1>h+w:
        x2 = w
        y2 = 2*h+w-val1
    else:
        x2 = val1-h
        y2 = h
    
    #val1_str = f"{val1:4d}" if val1 is not None else "    "
    #val2_str = f"{val2:4d}" if val2 is not None else "    "
    # Calculate the angle theta with the vertical
    #dx = x2 - x1
    #dy = y2 - y1
    #theta = np.degrees(np.arctan2(dx, dy))  # Angle in degrees
    #print(f"Start: ({x1},{y1}), End: ({x2},{y2})", f"theta: {theta:.2f} | after conversion - ", f"val1: {val1_str}, val2: {val2_str}")
    
    return (x1, y1), (x2, y2)

# Function to create a mask based on HSV values
def hsv_mask(image, bounds):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract lower and upper bounds from the dictionary
    lower_bound = np.array([bounds['h_low'], bounds['s_low'], bounds['v_low']])
    upper_bound = np.array([bounds['h_high'], bounds['s_high'], bounds['v_high']])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    return mask

# Function to get pixel coordinates when clicked and dragged
def get_points(event):
    global start_x, start_y, image, photo, pt1, pt2, val1, val2, mask

    if event.type == tk.EventType.ButtonPress:
        # Mouse button pressed (start drawing)
        start_x, start_y = event.x, event.y

    elif event.type == tk.EventType.Motion:
        # Mouse moved (draw line as the mouse moves)
        if start_x and start_y:
            # Draw the line on the image using OpenCV
            end_x, end_y = event.x, event.y

            # Find intersections with the edges
            intersections = find_edge_intersections(image.shape[1], image.shape[0], (start_x, start_y), (end_x, end_y))
            (x1, y1), (x2, y2) = intersections[0], intersections[1]

            val1, val2 = point_to_border(image.shape[1], image.shape[0], (x1, y1), (x2, y2))
            (x1, y1), (x2, y2) = border_to_point(image.shape[1], image.shape[0], val1, val2)

            pt1 = (x1, y1)
            pt2 = (x2, y2)

            update_mask()

    elif event.type == tk.EventType.ButtonRelease:
        image_copy = image.copy()
        # Mouse button released (end drawing)
        if start_x and start_y:
            # Draw the final line
            end_x, end_y = event.x, event.y
            
            # Find intersections with the edges
            intersections = find_edge_intersections(image.shape[1], image.shape[0], (start_x, start_y), (end_x, end_y))
            (x1, y1), (x2, y2) = intersections[0], intersections[1]

            val1, val2 = point_to_border(image.shape[1], image.shape[0], (x1, y1), (x2, y2))
            (x1, y1), (x2, y2) = border_to_point(image.shape[1], image.shape[0], val1, val2)

            pt1 = (x1, y1)
            pt2 = (x2, y2)

            update_mask()

            # Reset start points
            start_x, start_y = None, None

# Function to draw a line on the image
def draw_line(image, pt1, pt2):
    cv2.line(image, pt1, pt2, (255, 255, 255), 5)
    cv2.circle(image, pt1, 7, (255, 255, 255), -1)
    cv2.circle(image, pt2, 7, (255, 255, 255), -1)
    
    cv2.line(image, pt1, pt2, (0, 0, 0), 2)
    cv2.circle(image, pt1, 5, (0, 0, 255), -1)
    cv2.circle(image, pt2, 5, (0, 255, 0), -1)
    return image

# Function to update the canvas with the new image
def update_canvas(image=None):
    if image is not None:
        # Convert the OpenCV image to a format Tkinter can use
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image_pil)

        # Update the canvas with the new image
        canvas.image = photo
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)

# Function to convert the mask to class labels
def to_class(mask, threshold=128):
    b, g, r = cv2.split(mask)

    r_bin = (r > threshold).astype(np.uint8)
    g_bin = (g > threshold).astype(np.uint8)
    b_bin = (b > threshold).astype(np.uint8)

    class_map = np.zeros(mask.shape[:2], dtype=np.uint8)

    class_map[(r_bin == 0) & (g_bin == 0) & (b_bin == 1)] = 0  # Blue
    class_map[(r_bin == 1) & (g_bin == 0) & (b_bin == 0)] = 1  # Red
    class_map[(r_bin == 0) & (g_bin == 1) & (b_bin == 0)] = 2  # Green
    class_map[(r_bin == 1) & (g_bin == 0) & (b_bin == 1)] = 3  # Magenta
    class_map[(r_bin == 1) & (g_bin == 1) & (b_bin == 0)] = 4  # Yellow
    class_map[(r_bin == 0) & (g_bin == 1) & (b_bin == 1)] = 5  # Cyan
    class_map[(r_bin == 1) & (g_bin == 1) & (b_bin == 1)] = 6  # White

    return class_map

# Function to update the mask based on the selected points and HSV values
def update_mask():
    global image, pt1, pt2, mask_class, mask_bgr, line_mask
    x1, y1 = pt1
    x2, y2 = pt2
    image_copy = image.copy()

    if image is not None:
        # Recalculate the mask
        G1_mask = hsv_mask(image_copy, {
            'h_low': slider_vars['h_low1'].get(),
            'h_high': slider_vars['h_high1'].get(),
            's_low': slider_vars['s_low1'].get(),
            's_high': slider_vars['s_high1'].get(),
            'v_low': slider_vars['v_low1'].get(),
            'v_high': slider_vars['v_high1'].get()
        })
        G2_mask = hsv_mask(image_copy, {
            'h_low': slider_vars['h_low2'].get(),
            'h_high': slider_vars['h_high2'].get(),
            's_low': slider_vars['s_low2'].get(),
            's_high': slider_vars['s_high2'].get(),
            'v_low': slider_vars['v_low2'].get(),
            'v_high': slider_vars['v_high2'].get()
        })
        B1_mask = hsv_mask(image_copy, {
            'h_low': slider_vars['h_low3'].get(),
            'h_high': slider_vars['h_high3'].get(),
            's_low': slider_vars['s_low3'].get(),
            's_high': slider_vars['s_high3'].get(),
            'v_low': slider_vars['v_low3'].get(),
            'v_high': slider_vars['v_high3'].get()
        })
        B2_mask = hsv_mask(image_copy, {
            'h_low': slider_vars['h_low4'].get(),
            'h_high': slider_vars['h_high4'].get(),
            's_low': slider_vars['s_low4'].get(),
            's_high': slider_vars['s_high4'].get(),
            'v_low': slider_vars['v_low4'].get(),
            'v_high': slider_vars['v_high4'].get()
        })
        R1_mask = hsv_mask(image_copy, {
            'h_low': slider_vars['h_low5'].get(),
            'h_high': slider_vars['h_high5'].get(),
            's_low': slider_vars['s_low5'].get(),
            's_high': slider_vars['s_high5'].get(),
            'v_low': slider_vars['v_low5'].get(),
            'v_high': slider_vars['v_high5'].get()
        })
        R2_mask = hsv_mask(image_copy, {
            'h_low': slider_vars['h_low6'].get(),
            'h_high': slider_vars['h_high6'].get(),
            's_low': slider_vars['s_low6'].get(),
            's_high': slider_vars['s_high6'].get(),
            'v_low': slider_vars['v_low6'].get(),
            'v_high': slider_vars['v_high6'].get()
        })
        
        # Combine the masks
        green_mask = cv2.bitwise_or(G1_mask, G2_mask)
        blue_mask = cv2.bitwise_or(B1_mask, B2_mask)
        red_mask = cv2.bitwise_or(R1_mask, R2_mask)

        #green_mask = keep_largest_components(green_mask, 8)

        # Setup masks priorities
        blue_mask = np.where(green_mask != 0, 0, blue_mask)
        blue_mask = np.where(red_mask != 0, 0, blue_mask)
        red_mask = np.where(green_mask != 0, 0, red_mask)

        mask_bgr = cv2.merge((blue_mask, green_mask, red_mask))
        mask_class = to_class(mask_bgr, 128)

        line_mask = create_line_mask(image, pt1, pt2, int(line_width_entry.get()), int(angle_offset_entry.get()))
        line_mask = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)

        try:
            image_alpha = float(image_blend_entry.get())
            mask_alpha = float(rgb_blend_entry.get())
            line_alpha = float(line_blend_entry.get())
        except ValueError:
            print("Error: Invalid input for alpha values. Using default values.")
            image_alpha = 0.5
            mask_alpha = 0.5
            line_alpha = 0.5

        image_mixed = cv2.addWeighted(image_copy, image_alpha, mask_bgr, mask_alpha, 0)

        image_mixed = cv2.addWeighted(image_mixed, image_alpha, line_mask*255, line_alpha, 0)

        image_mixed = draw_line(image_mixed, pt1, pt2)
        cv2.circle(image_mixed, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(image_mixed, (x2, y2), 5, (0, 255, 0), -1)

        #cv2.rectangle(red_mask, (x1, y1), (x2, y2), (255), -1)
        #cv2.fillPoly(red_mask, np.array([[(x1-80, y1), (x1+80, y1), (x2+200, y2), (x2-200, y2)]]), 255)

        #uniques = np.unique(mask_class, return_counts=True)
        #cv2.putText(image_mixed, f"Unique values: {uniques[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        #cv2.putText(image_mixed, f"Unique values: {uniques[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        root.focus()
        update_canvas(image_mixed)

# Function to apply CLAHE to the BGR image
def clahe_bgr(img, clipLimit=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result

# Function to boost saturation
def boost_saturation(img, factor=1.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    hsv_boosted = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

# Function to adjust brightness and contrast
def brightness_contrast(img, alpha=1.2, beta=20):
    # alpha = contrast [1.0–3.0], beta = brightness [0–100]
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Function to shift the hue
def shift_hue(image, shift_value):
    """
    Shift the hue of an image in the HSV color space, wrapping around the hue circle.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        shift_value (int): Amount to shift the hue, in degrees (0-180 for OpenCV).

    Returns:
        numpy.ndarray: Image with shifted hue, in BGR format.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the hue channel
    h, s, v = cv2.split(hsv)

    # Shift the hue channel and wrap around using modulo
    h = (h.astype(int) + shift_value) % 180  # OpenCV uses 0-179 for hue
    h = h.astype(np.uint8)

    # Merge the channels back
    hsv_shifted = cv2.merge((h, s, v))

    # Convert back to BGR color space
    shifted_image = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

    return shifted_image

# Function to read an image and apply enhancements
def read_image(image_path):
    """Read an image and apply CLAHE and saturation boost."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    img = clahe_bgr(img, clipLimit=0.25)
    img = brightness_contrast(img, alpha=1.25, beta=-40)
    img = boost_saturation(img, factor=1.4)
    #img = shift_hue(img, 0)

    return img

# Function to display the next image
def display_next_image(index):
    global image, photo

    if index >= len(image_files):
        print("No more images.")
        root.quit()
        return

    # Load the next image
    image_path = os.path.join(dataset_folder, image_files[index])
    #image = cv2.imread(image_path)
    image = read_image(image_path)
    print(f"Loaded image: {image_path}")
    
    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image to a format Tkinter can use
    image_pil = Image.fromarray(image_rgb)
    canvas.config(width=image_pil.width, height=image_pil.height)
    photo = ImageTk.PhotoImage(image_pil)

    # Update the canvas with the new image
    canvas.image = photo
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    update_mask()

# Function to save slider values and current image index to a file
def save_settings():
    global current_index
    settings = {
        "current_index": current_index,
        "image_blend": image_blend_entry.get(),
        "rgb_blend": rgb_blend_entry.get(),
        "line_blend": line_blend_entry.get(),
        "line_width": line_width_entry.get(),
        "angle_offset": angle_offset_entry.get(),
        "image_class": selection_var.get(),
        "sliders": [
            {
                "h_low": slider_vars[f"h_low{i+1}"].get(),
                "h_high": slider_vars[f"h_high{i+1}"].get(),
                "s_low": slider_vars[f"s_low{i+1}"].get(),
                "s_high": slider_vars[f"s_high{i+1}"].get(),
                "v_low": slider_vars[f"v_low{i+1}"].get(),
                "v_high": slider_vars[f"v_high{i+1}"].get(),
            }
            for i in range(6)
        ]
    }
    with open(f"{dataset_folder}.json", "w") as f:
        json.dump(settings, f)
    print("Settings saved.")

# Function to load slider values and current image index from a file
def load_settings():
    global current_index
    try:
        with open(f"{dataset_folder}.json", "r") as f:
            settings = json.load(f)
            current_index = settings.get("current_index", 0)

            # Load blend and config entries
            image_blend_entry.delete(0, tk.END)
            image_blend_entry.insert(0, settings.get("image_blend", "0.5"))

            rgb_blend_entry.delete(0, tk.END)
            rgb_blend_entry.insert(0, settings.get("rgb_blend", "0.5"))

            line_blend_entry.delete(0, tk.END)
            line_blend_entry.insert(0, settings.get("line_blend", "0.5"))

            line_width_entry.delete(0, tk.END)
            line_width_entry.insert(0, settings.get("line_width", "200"))

            angle_offset_entry.delete(0, tk.END)
            angle_offset_entry.insert(0, settings.get("angle_offset", "15"))

            # Set the selected image class (radiobutton)
            selection_var.set(settings.get("image_class", "Headland"))

            # Load sliders
            sliders = settings.get("sliders", [])

            if len(sliders) != 6:
                print(f"Warning: Expected 6 slider sets, but found {len(sliders)}.")
                while len(sliders) < 6:
                    sliders.append({
                        "h_low": 45, "h_high": 48,
                        "s_low": 130, "s_high": 169,
                        "v_low": 68, "v_high": 255
                    })

            for i in range(6):
                slider_vars[f"h_low{i+1}"].set(sliders[i]["h_low"])
                slider_vars[f"h_high{i+1}"].set(sliders[i]["h_high"])
                slider_vars[f"s_low{i+1}"].set(sliders[i]["s_low"])
                slider_vars[f"s_high{i+1}"].set(sliders[i]["s_high"])
                slider_vars[f"v_low{i+1}"].set(sliders[i]["v_low"])
                slider_vars[f"v_high{i+1}"].set(sliders[i]["v_high"])

            print("Settings loaded.")
            display_next_image(current_index)

    except FileNotFoundError:
        print("No settings file found.")
    except KeyError as e:
        print(f"Error: Missing key in settings file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Function to open a new window for sliders
def open_slider_window():
    global slider_vars
    slider_window = tk.Toplevel(root)
    slider_window.title("Slider Controls")

    # Create six sets of sliders
    for i in range(6):
        frame = tk.Frame(slider_window)
        frame.pack(pady=5)

        tk.Label(frame, text=f"Set {i+1} - H Low").grid(row=0, column=0)
        tk.Scale(frame, from_=0, to=255, orient="horizontal", length=300, variable=slider_vars[f"h_low{i+1}"], command=lambda _: update_mask()).grid(row=0, column=1)

        tk.Label(frame, text="H High").grid(row=0, column=2)
        tk.Scale(frame, from_=0, to=255, orient="horizontal", length=300, variable=slider_vars[f"h_high{i+1}"], command=lambda _: update_mask()).grid(row=0, column=3)

        tk.Label(frame, text="S Low").grid(row=1, column=0)
        tk.Scale(frame, from_=0, to=255, orient="horizontal", length=300, variable=slider_vars[f"s_low{i+1}"], command=lambda _: update_mask()).grid(row=1, column=1)

        tk.Label(frame, text="S High").grid(row=1, column=2)
        tk.Scale(frame, from_=0, to=255, orient="horizontal", length=300, variable=slider_vars[f"s_high{i+1}"], command=lambda _: update_mask()).grid(row=1, column=3)

        tk.Label(frame, text="V Low").grid(row=2, column=0)
        tk.Scale(frame, from_=0, to=255, orient="horizontal", length=300, variable=slider_vars[f"v_low{i+1}"], command=lambda _: update_mask()).grid(row=2, column=1)

        tk.Label(frame, text="V High").grid(row=2, column=2)
        tk.Scale(frame, from_=0, to=255, orient="horizontal", length=300, variable=slider_vars[f"v_high{i+1}"], command=lambda _: update_mask()).grid(row=2, column=3)
        # Add save, load, and exit buttons at the bottom
    button_frame = tk.Frame(slider_window)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(button_frame, text="Load", command=load_settings).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(button_frame, text="Exit", command=slider_window.destroy).pack(side=tk.LEFT, padx=5, pady=5)

# Function to handle the slider window closing
def save_mask(current_index, mask):
    mask_path = os.path.join(dataset_folder, f"{image_files[current_index].split('img')[0]}mask.png")
    cv2.imwrite(mask_path, mask)

# Save the mask as a PNG file
def save_mask_class(current_index, mask_class):
    mask_path = os.path.join(dataset_folder, f"{image_files[current_index].split('img')[0]}class.png")
    cv2.imwrite(mask_path, mask_class)

# Save line mask as a PNG file
def save_line_mask(current_index, line_mask):
    line_path = os.path.join(dataset_folder, f"{image_files[current_index].split('img')[0]}line.png")
    cv2.imwrite(line_path, line_mask)

# Save the line data as a JSON file
def save_line(current_index, pt1, pt2):
    line_path = os.path.join(dataset_folder, f"{image_files[current_index].split('img')[0]}line.json")
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    theta = np.degrees(np.arctan2(dx, dy))
    
    data = {
        "pt1": pt1,
        "pt2": pt2,
        "theta": theta,
        "val1": val1,
        "val2": val2,
        "class": selection_var.get(),
    }
    with open(line_path, 'w') as f:
        json.dump(data, f)
    print(data)
    print(line_path)
    print(f"Final point: ({x2},{y2}) | Angle: {theta:.2f} | Val1: {val1}, Val2: {val2}")

# Function to create a line mask
def create_line_mask(image, pt1, pt2, L_width=100, angle_offset=0):
    # Create a mask with the same size as the image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    L_length = image.shape[1]
    delta = np.radians(angle_offset)

    theta = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

    # Calculate the center of the line
    center_x = (pt1[0] + pt2[0]) // 2
    center_y = (pt1[1] + pt2[1]) // 2

    X_R_base = center_x + L_width * np.cos(theta+np.radians(90))
    Y_R_base = center_y + L_width * np.sin(theta+np.radians(90))
    X_L_base = center_x + L_width * np.cos(theta-np.radians(90))
    Y_L_base = center_y + L_width * np.sin(theta-np.radians(90))

    X_L_S = X_L_base - L_length * np.cos(theta-np.sin(theta)*delta)
    Y_L_S = Y_L_base - L_length * np.sin(theta-np.sin(theta)*delta)
    X_L_E = X_L_base + L_length * np.cos(theta-np.sin(theta)*delta)
    Y_L_E = Y_L_base + L_length * np.sin(theta-np.sin(theta)*delta)

    X_R_S = X_R_base - L_length * np.cos(theta+np.sin(theta)*delta)
    Y_R_S = Y_R_base - L_length * np.sin(theta+np.sin(theta)*delta)
    X_R_E = X_R_base + L_length * np.cos(theta+np.sin(theta)*delta)
    Y_R_E = Y_R_base + L_length * np.sin(theta+np.sin(theta)*delta)

    # Draw the lines on the mask
    #cv2.line(mask, (int(X_R_S), int(Y_R_S)), (int(X_R_E), int(Y_R_E)), (255), thickness=2)
    #cv2.line(mask, (int(X_L_S), int(Y_L_S)), (int(X_L_E), int(Y_L_E)), (255), thickness=2)
    #cv2.line(mask, pt1, pt2, (128), thickness=5)

    cv2.fillPoly(mask, np.array([[(int(X_R_S), int(Y_R_S)), (int(X_R_E), int(Y_R_E)), (int(X_L_E), int(Y_L_E)), (int(X_L_S), int(Y_L_S))]]), 1)
    return mask

# Create a Tkinter window
root = tk.Tk()

# Set up the canvas for handling click events
canvas = tk.Canvas(root)
canvas.pack()

# Bind mouse events
canvas.bind("<ButtonPress-1>", get_points)
canvas.bind("<B1-Motion>", get_points)
canvas.bind("<ButtonRelease-1>", get_points)

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Line Width
line_width_label = tk.Label(input_frame, text="Line Width:")
line_width_label.grid(row=0, column=0, padx=3, pady=5)
line_width_entry = tk.Entry(input_frame, width=8)
line_width_entry.insert(0, "200")
line_width_entry.grid(row=0, column=1, padx=3, pady=5)

# Angle Offset
angle_offset_label = tk.Label(input_frame, text="Angle Offset:")
angle_offset_label.grid(row=0, column=2, padx=3, pady=5)
angle_offset_entry = tk.Entry(input_frame, width=8)
angle_offset_entry.insert(0, "15")
angle_offset_entry.grid(row=0, column=3, padx=3, pady=5)

# Image Blend
image_blend_label = tk.Label(input_frame, text="Image alpha [0-1]:")
image_blend_label.grid(row=0, column=4, padx=3, pady=5)
image_blend_entry = tk.Entry(input_frame, width=6)
image_blend_entry.insert(0, "0.5")
image_blend_entry.grid(row=0, column=5, padx=3, pady=5)

# RGB Mask Blend
rgb_blend_label = tk.Label(input_frame, text="Mask alpha [0-1]:")
rgb_blend_label.grid(row=0, column=6, padx=3, pady=5)
rgb_blend_entry = tk.Entry(input_frame, width=6)
rgb_blend_entry.insert(0, "0.5")
rgb_blend_entry.grid(row=0, column=7, padx=3, pady=5)

# Line Blend
line_blend_label = tk.Label(input_frame, text="Line alpha [0-1]:")
line_blend_label.grid(row=0, column=8, padx=3, pady=5)
line_blend_entry = tk.Entry(input_frame, width=6)
line_blend_entry.insert(0, "0.5")
line_blend_entry.grid(row=0, column=9, padx=3, pady=5)

line_width_entry.bind("<Return>", lambda e: update_mask())
angle_offset_entry.bind("<Return>", lambda e: update_mask())
image_blend_entry.bind("<Return>", lambda e: update_mask())
rgb_blend_entry.bind("<Return>", lambda e: update_mask())
line_blend_entry.bind("<Return>", lambda e: update_mask())

# ========== Bottom Frame for Buttons and Radios ==========
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=10, fill="x")

# Left side: Open Slider Controls button
open_button = tk.Button(bottom_frame, text="Open Slider Controls", command=open_slider_window)
open_button.pack(side=tk.LEFT, padx=10)

# Center: Radiobuttons
selection_var = tk.StringVar(value="Headland")
options = ["Headland", "InWay", "InRow", "OutWay", "Lost"]

radio_frame = tk.Frame(bottom_frame)
radio_frame.pack(side=tk.LEFT, expand=True)

for option in options:
    tk.Radiobutton(radio_frame, text=option, variable=selection_var, value=option).pack(side=tk.LEFT, padx=5)

# Right side: Save, Load, Exit buttons
action_frame = tk.Frame(bottom_frame)
action_frame.pack(side=tk.RIGHT, padx=10)

save_button = tk.Button(action_frame, text="Save", width=8, command=save_settings)
save_button.pack(side=tk.LEFT, padx=5)

load_button = tk.Button(action_frame, text="Load", width=8, command=load_settings)
load_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(action_frame, text="Exit", width=8, command=root.destroy)
exit_button.pack(side=tk.LEFT, padx=5)

# Green 1
slider_vars[f"h_low1"]  = tk.IntVar(value=10)
slider_vars[f"h_high1"] = tk.IntVar(value=80)
slider_vars[f"s_low1"]  = tk.IntVar(value=0)
slider_vars[f"s_high1"] = tk.IntVar(value=255)
slider_vars[f"v_low1"]  = tk.IntVar(value=0)
slider_vars[f"v_high1"] = tk.IntVar(value=255)
# Green 2
slider_vars[f"h_low2"]  = tk.IntVar(value=0)
slider_vars[f"h_high2"] = tk.IntVar(value=0)
slider_vars[f"s_low2"]  = tk.IntVar(value=0)
slider_vars[f"s_high2"] = tk.IntVar(value=0)
slider_vars[f"v_low2"]  = tk.IntVar(value=0)
slider_vars[f"v_high2"] = tk.IntVar(value=0)
# Blue 1
slider_vars[f"h_low3"]  = tk.IntVar(value=81)
slider_vars[f"h_high3"] = tk.IntVar(value=120)
slider_vars[f"s_low3"]  = tk.IntVar(value=0)
slider_vars[f"s_high3"] = tk.IntVar(value=255)
slider_vars[f"v_low3"]  = tk.IntVar(value=0)
slider_vars[f"v_high3"] = tk.IntVar(value=255)
# Blue 2
slider_vars[f"h_low4"]  = tk.IntVar(value=0)
slider_vars[f"h_high4"] = tk.IntVar(value=0)
slider_vars[f"s_low4"]  = tk.IntVar(value=0)
slider_vars[f"s_high4"] = tk.IntVar(value=0)
slider_vars[f"v_low4"]  = tk.IntVar(value=0)
slider_vars[f"v_high4"] = tk.IntVar(value=0)
# Red 1
slider_vars[f"h_low5"]  = tk.IntVar(value=121)
slider_vars[f"h_high5"] = tk.IntVar(value=180)
slider_vars[f"s_low5"]  = tk.IntVar(value=0)
slider_vars[f"s_high5"] = tk.IntVar(value=255)
slider_vars[f"v_low5"]  = tk.IntVar(value=0)
slider_vars[f"v_high5"] = tk.IntVar(value=255)
# Red 2
slider_vars[f"h_low6"]  = tk.IntVar(value=0)
slider_vars[f"h_high6"] = tk.IntVar(value=10)
slider_vars[f"s_low6"]  = tk.IntVar(value=0)
slider_vars[f"s_high6"] = tk.IntVar(value=255)
slider_vars[f"v_low6"]  = tk.IntVar(value=0)
slider_vars[f"v_high6"] = tk.IntVar(value=255)

#open_slider_window()

# Start displaying the first image
if image_files:
    current_index = 0
    print(f"Displaying image {current_index + 1} of {len(image_files)}")
    display_next_image(current_index)

    def handle_keypress(event):
        global current_index
        if event.char == 'w':
            # Save the mask and line data
            save_mask(current_index, mask_bgr)
            save_mask_class(current_index, mask_class)
            save_line(current_index, pt1, pt2)
            save_line_mask(current_index, line_mask)
            print("Saved!")
            current_index += 1

            if current_index < len(image_files):
                print(f"Displaying image {current_index + 1} of {len(image_files)}")
                display_next_image(current_index)
                update_mask()
            else:
                current_index = 0
                print("Last Image. Looping back to first.")
                print(f"Displaying image {current_index + 1} of {len(image_files)}")
                display_next_image(current_index)
                update_mask()
        elif event.char == 'e':
            current_index += 1
            if current_index < len(image_files):
                print(f"Displaying image {current_index + 1} of {len(image_files)}")
                display_next_image(current_index)
                update_mask()
            else:
                current_index = 0
                print("Last Image. Looping back to first.")
                print(f"Displaying image {current_index + 1} of {len(image_files)}")
                display_next_image(current_index)
                update_mask()

        elif event.char == 's':
            current_index -= 1
            if current_index >= 0:
                print(f"Displaying image {current_index + 1} of {len(image_files)}")
                display_next_image(current_index)
                update_mask()
            else:
                current_index = len(image_files) - 1
                print("First Image. Looping back to last.")
                print(f"Displaying image {current_index + 1} of {len(image_files)}")
                display_next_image(current_index)
                update_mask()
            
        elif event.char == 'q':
            root.quit()

    root.bind("<Key>", handle_keypress)
else:
    print("No images found in the folder.")
    root.quit()

root.mainloop()