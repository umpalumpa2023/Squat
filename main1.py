from PIL import Image, ImageTk 
import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time
from collections import deque
import winsound

# ==========================
# Global Variables
# ==========================
squat_count = 0
measurement_active = False
sound_enabled = False
data_storage = {'femur_angle': [], 'knee_angle': [], 'timestamps': []}
cap = None  # Global cap variable for camera
rep_state = "up"  # Used to track whether the squat is in the 'up' or 'down' phase

# Initialize ArUco Dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()

# Initialize storage for moving average
hip_angle_history = deque(maxlen=5)
knee_angle_history = deque(maxlen=5)

# Last valid marker positions
last_valid_marker_positions = {}

# Initialize frame counters for each marker
marker_timeout_frames = {1: 0, 12: 0, 123: 0}  # To count frames since each marker was last detected
max_timeout = 15  # Number of frames to keep the last position before removing it

# Define the sampling rate (frames per second)
fps = 30  # Adjust this value according to your actual  data sampling rate

# ==========================
# Functions for Angle Calculation
# ==========================
def calculate_angles(marker_positions):
    if 1 in marker_positions and 12 in marker_positions and 123 in marker_positions:
        hip_pos = marker_positions[1]
        knee_pos = marker_positions[12]
        ankle_pos = marker_positions[123]

        vec_hip_knee = np.array(knee_pos) - np.array(hip_pos)
        vec_knee_ankle = np.array(ankle_pos) - np.array(knee_pos)

        cos_theta = np.dot(vec_hip_knee, vec_knee_ankle) / (np.linalg.norm(vec_hip_knee) * np.linalg.norm(vec_knee_ankle))
        knee_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        femur_angle = np.degrees(np.arctan2(vec_hip_knee[1], vec_hip_knee[0])) - 90
        if femur_angle < 0:
            femur_angle = -femur_angle
        femur_angle = np.clip(femur_angle, 0, 180)

        # Store angles in history and calculate moving average
        hip_angle_history.append(femur_angle)
        knee_angle_history.append(knee_angle)

        femur_angle_avg = np.mean(hip_angle_history)
        knee_angle_avg = np.mean(knee_angle_history)

        return femur_angle_avg, knee_angle_avg
    return None, None

# ==========================
# Functions for Marker Detection
# ==========================
def find_markers(frame):
    global marker_timeout_frames
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    marker_positions = {}

    # Reset frame counters for detected markers and update their positions
    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id in [1, 12, 123]:  # Relevant markers
                position = np.mean(corner[0], axis=0)
                marker_positions[marker_id] = position
                last_valid_marker_positions[marker_id] = position  # Update last valid position
                marker_timeout_frames[marker_id] = 0  # Reset timeout for this marker
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # Increase timeout counters for markers that were not detected
    for marker_id in [1, 12, 123]:
        if marker_id not in marker_positions:
            marker_timeout_frames[marker_id] += 1  # Increment timeout counter
            # Only keep the last valid position if the marker hasn't timed out
            if marker_timeout_frames[marker_id] <= max_timeout and marker_id in last_valid_marker_positions:
                marker_positions[marker_id] = last_valid_marker_positions[marker_id]
        else:
            marker_timeout_frames[marker_id] = 0  # Reset if detected

    # Draw lines between markers only if they are currently detected or within the timeout
    if 1 in marker_positions and marker_positions[1] is not None and \
       12 in marker_positions and marker_positions[12] is not None:
        hip_pos = tuple(map(int, marker_positions[1]))  # Hip position
        knee_pos = tuple(map(int, marker_positions[12]))  # Knee position
        cv2.line(frame, hip_pos, knee_pos, (0, 255, 0), 2)  # Line between hip and knee (green)

    if 12 in marker_positions and marker_positions[12] is not None and \
       123 in marker_positions and marker_positions[123] is not None:
        knee_pos = tuple(map(int, marker_positions[12]))  # Knee position
        ankle_pos = tuple(map(int, marker_positions[123]))  # Ankle position
        cv2.line(frame, knee_pos, ankle_pos, (0, 0, 255), 2)  # Line between knee and ankle (red)

    return marker_positions, frame



# ==========================
# Functions for Measurement Loop
# ==========================
def measurement_loop():
    global measurement_active, data_storage, squat_count, sound_enabled, rep_state

    if not measurement_active:
        return
    
    ret, frame = cap.read()
    if ret:
        marker_positions, frame = find_markers(frame)
        femur_angle, knee_angle = calculate_angles(marker_positions)
        update_marker_status(marker_positions)

        if femur_angle is not None:
            current_time = time.time()
            data_storage['femur_angle'].append(femur_angle)  # Store femur angle for later analysis
            data_storage['knee_angle'].append(knee_angle)  # Store knee angle for later analysis
            data_storage['timestamps'].append(current_time)

            # Debugging: Output current values for monitoring
            print(f"Femur Angle: {femur_angle}, Knee Angle: {knee_angle}, Rep State: {rep_state}, Squat Count: {squat_count}")

            # Counting logic based on femur angle relative to the bottom frame edge
            if femur_angle <= 90:  # If the thigh is parallel or lower than the horizontal axis
                if rep_state == "up":  # If previously in the "up" position
                    rep_state = "down"  # Switch to "down"
                    print("Switch to down state")

            elif femur_angle > 90:  # If the thigh is significantly above the horizontal axis again
                if rep_state == "down":  # If previously in the "down" position
                    squat_count += 1  # Count a repetition
                    rep_state = "up"  # Switch back to "up"
                    update_squat_count_label()
                    print(f"Squat counted! New count: {squat_count}")
                    if sound_enabled:
                        winsound.Beep(1000, 200)  # Beep at 1000 Hz for 200 ms
            update_visualization()

        # Convert the frame to a Tkinter-compatible format and update the label
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(img)  # Convert to PIL Image
        imgtk = ImageTk.PhotoImage(image=img)  # Convert to ImageTk format
        camera_label.imgtk = imgtk  # Keep reference to prevent garbage collection
        camera_label.config(image=imgtk)  # Update label

    root.after(10, measurement_loop)

# ==========================
# Functions for GUI Updates
# ==========================
def update_marker_status(marker_positions):
    hip_status.config(text="Hüfte: Erkannt" if 1 in marker_positions else "Hüfte: Nicht erkannt")
    knee_status.config(text="Knie: Erkannt" if 12 in marker_positions else "Knie: Nicht erkannt")
    ankle_status.config(text="Sprunggelenk: Erkannt" if 123 in marker_positions else "Sprunggelenk: Nicht erkannt")

    if 1 in marker_positions and 12 in marker_positions and 123 in marker_positions:
        all_markers_status.config(text="Alle Marker erkannt!", foreground="green")
    else:
        all_markers_status.config(text="Warte auf Marker...", foreground="red")

def update_squat_count_label():
    squat_count_label.config(text=f"Squat Count: {squat_count}")

def update_visualization():
    # Calculate the number of data points to display for the last 5 seconds
    points_to_display = 5 * fps

    '''    
    # Update Hip Angle plot with the last 5 seconds of data
    hip_angle_line.set_ydata(data_storage['femur_angle'][-points_to_display:])
    hip_angle_line.set_xdata(range(len(data_storage['femur_angle'][-points_to_display:])))
    '''
    
    # Update Knee Angle plot with the last 5 seconds of data
    knee_angle_line.set_ydata(data_storage['knee_angle'][-points_to_display:])
    knee_angle_line.set_xdata(range(len(data_storage['knee_angle'][-points_to_display:])))
    
    # Refresh each subplot
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    # Redraw the canvas with updated data
    canvas.draw()

    # Update angle labels
    if data_storage['femur_angle']:
        femur_angle_label.config(text=f"Femur Angle: {data_storage['femur_angle'][-1]:.2f}°")
    if data_storage['knee_angle']:
        knee_angle_label.config(text=f"Knee Angle: {data_storage['knee_angle'][-1]:.2f}°")

# ==========================
# Functions for GUI Controls
# ==========================
def start_measurement():
    global measurement_active, cap
    measurement_active = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
    
    # Clear the camera_label by setting it to an empty image
    camera_label.config(image='')  # This will effectively "refresh" the GUI to look empty

def reset_counter():
    global squat_count
    squat_count = 0
    update_squat_count_label()

def toggle_sound():
    global sound_enabled
    sound_enabled = not sound_enabled

# ==========================
# GUI Initialization
# ==========================
root = tk.Tk()
root.title("Squat Measurement GUI")

# Configure the root window to dynamically resize widgets
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure the frame to resize with window
for i in range(10):  # There are 10 rows before the camera label and plot
    frame.rowconfigure(i, weight=1)
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

start_button = ttk.Button(frame, text="Start", command=start_measurement)
start_button.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

stop_button = ttk.Button(frame, text="Stop", command=stop_measurement)
stop_button.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

squat_count_label = ttk.Label(frame, text=f"Squat Count: {squat_count}")
squat_count_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

reset_button = ttk.Button(frame, text="Reset Counter", command=reset_counter)
reset_button.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

sound_checkbox = ttk.Checkbutton(frame, text="Enable Sound", command=toggle_sound)
sound_checkbox.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))

hip_status = ttk.Label(frame, text="Hüfte: Nicht erkannt")
hip_status.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

knee_status = ttk.Label(frame, text="Knie: Nicht erkannt")
knee_status.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

ankle_status = ttk.Label(frame, text="Sprunggelenk: Nicht erkannt")
ankle_status.grid(row=6, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

all_markers_status = ttk.Label(frame, text="Warte auf Marker...", foreground="red")
all_markers_status.grid(row=7, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

femur_angle_label = ttk.Label(frame, text="Femur Angle: N/A")
femur_angle_label.grid(row=8, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

knee_angle_label = ttk.Label(frame, text="Knee Angle: N/A")
knee_angle_label.grid(row=9, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Label to display the camera frame
camera_label = tk.Label(root)
camera_label.grid(row=10, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

# Matplotlib Figure for Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))

'''
# Configure the first subplot for femur angle
ax1.set_ylim(0, 260)
ax1.set_xlim(0, 150)
ax1.set_ylabel('Femur Angle (degrees)')
ax1.set_xlabel('Time (frames)')
hip_angle_line, = ax1.plot([], [], label='Femur Angle', color='blue')
ax1.legend()
'''

# Configure the second subplot for knee angle
ax2.set_ylim(0, 260)
ax2.set_xlim(0, 150)
ax2.set_ylabel('Knee Angle (degrees)')
ax2.set_xlabel('Time (frames)')
knee_angle_line, = ax2.plot([], [], label='Knee Angle', color='red')
ax2.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure row and column weights to allow dynamic resizing
root.rowconfigure(10, weight=1)  # Camera frame row
root.rowconfigure(11, weight=1)  # Plot row
root.columnconfigure(0, weight=1)

root.mainloop()
