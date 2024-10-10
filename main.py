import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time

# Global Variables
squat_count = 0
measurement_active = False
sound_enabled = False
data_storage = {'knee_angle': [], 'femur_angle': [], 'timestamps': []}
cap = None  # Global cap variable for camera

# Initialize ArUco Dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# Generate a marker (Incorporated from Notebook)
marker_id = 42
marker_size = 200  # Size in pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
cv2.imwrite('/mnt/data/marker_42.png', marker_image)

# Display the generated marker using matplotlib (Incorporated from Notebook)
plt.imshow(marker_image, cmap='gray', interpolation='nearest')
plt.axis('off')  # Hide axes
plt.title(f'ArUco Marker {marker_id}')
plt.show()

# Function to calculate angles based on ArUco marker positions
def calculate_angles(marker_positions):
    if len(marker_positions) >= 2:
        # Assume markers 0 and 1 are being used for the femur and knee
        point1 = marker_positions[0][0]
        point2 = marker_positions[1][0]

        # Calculate the angle in degrees between the two points relative to the horizontal
        delta_y = point2[1] - point1[1]
        delta_x = point2[0] - point1[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        femur_angle = angle
        knee_angle = 180 - angle  # Simplified for demo purposes

        return femur_angle, knee_angle
    return None, None

# Function to detect markers
def find_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    marker_positions = []

    if ids is not None:
        for corner in corners:
            position = np.mean(corner[0], axis=0)  # Average position of marker corners
            marker_positions.append(position)
    
    return marker_positions

# Function to handle measurement loop using after()
def measurement_loop():
    global measurement_active, data_storage, squat_count, sound_enabled

    if not measurement_active:
        return  # Stop if measurement is not active
    
    ret, frame = cap.read()
    if ret:
        # Find markers and calculate angles
        marker_positions = find_markers(frame)
        femur_angle, knee_angle = calculate_angles(marker_positions)

        if femur_angle is not None and knee_angle is not None:
            current_time = time.time()
            data_storage['knee_angle'].append(knee_angle)
            data_storage['femur_angle'].append(femur_angle)
            data_storage['timestamps'].append(current_time)

            # Check if valid squat (example condition)
            if femur_angle < 90:  # Simplified condition for demo
                squat_count += 1
                update_squat_count_label()
                if sound_enabled:
                    print("Beep!")  # Placeholder for actual sound

            # Update GUI visualization
            update_visualization()

        # Display the frame with detected markers
        cv2.imshow('Measurement Frame', frame)

    # Schedule the next frame capture
    root.after(10, measurement_loop)

# GUI Callbacks
def start_measurement():
    global measurement_active, cap
    measurement_active = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    measurement_loop()

def stop_measurement():
    global measurement_active, cap
    measurement_active = False
    if cap:
        cap.release()
        cv2.destroyAllWindows()

def reset_counter():
    global squat_count
    squat_count = 0
    update_squat_count_label()

def toggle_sound():
    global sound_enabled
    sound_enabled = not sound_enabled

# GUI Updates
def update_squat_count_label():
    squat_count_label.config(text=f"Squat Count: {squat_count}")

def update_visualization():
    femur_angle_line.set_ydata(data_storage['femur_angle'][-10:])
    knee_angle_line.set_ydata(data_storage['knee_angle'][-10:])
    ax.relim()
    ax.autoscale_view()
    canvas.draw()

# Initialize GUI
root = tk.Tk()
root.title("Squat Measurement GUI")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Start and Stop Buttons
start_button = ttk.Button(frame, text="Start", command=start_measurement)
start_button.grid(row=0, column=0, padx=5, pady=5)

stop_button = ttk.Button(frame, text="Stop", command=stop_measurement)
stop_button.grid(row=0, column=1, padx=5, pady=5)

# Squat Counter Label and Reset Button
squat_count_label = ttk.Label(frame, text=f"Squat Count: {squat_count}")
squat_count_label.grid(row=1, column=0, columnspan=2)

reset_button = ttk.Button(frame, text="Reset Counter", command=reset_counter)
reset_button.grid(row=2, column=0, columnspan=2, pady=5)

# Sound Checkbox
sound_checkbox = ttk.Checkbutton(frame, text="Enable Sound", command=toggle_sound)
sound_checkbox.grid(row=3, column=0, columnspan=2)

# Matplotlib Figure for Visualization
fig, ax = plt.subplots(figsize=(5, 2))
ax.set_ylim(0, 180)
ax.set_xlim(0, 10)  # Last 10 data points
ax.set_ylabel('Angle (degrees)')
ax.set_xlabel('Time (frames)')
femur_angle_line, = ax.plot([], [], label='Femur Angle', color='blue')
knee_angle_line, = ax.plot([], [], label='Knee Angle', color='red')
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=4, column=0, columnspan=2)

# Run the GUI
root.mainloop()
