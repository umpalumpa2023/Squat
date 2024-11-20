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

# Counter to track the number of successful squat repetitions.
squat_count = 0

# Boolean flag to determine if the measurement process is active.
measurement_active = False

# Boolean flag to toggle sound notifications (e.g., for feedback on squats).
sound_enabled = False

# Dictionary to store angle data and timestamps during measurement.
data_storage = {'femur_angle': [], 'knee_angle': [], 'timestamps': []}

# Global variable for the camera capture object (initialized later when accessing the camera).
cap = None

# Tracks the current state of the squat ('up' or 'down'), used for repetition detection.
rep_state = "up"

# Initialize the ArUco marker dictionary for marker detection.
# DICT_4X4_250: ArUco marker dictionary containing 4x4 binary patterns with 250 unique markers.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Detector parameters for ArUco marker detection.
aruco_params = cv2.aruco.DetectorParameters()

# Deque (double-ended queue) to store the recent history of femur angles for smoothing using moving averages.
femur_angle_history = deque(maxlen=5)

# Deque to store the recent history of knee angles for smoothing using moving averages.
knee_angle_history = deque(maxlen=5)

# Dictionary to store the last valid detected positions of markers (identified by marker IDs).
last_valid_marker_positions = {}

# Dictionary to track the number of frames since each marker was last detected.
# Marker IDs are keys, and values are frame counters.
marker_timeout_frames = {1: 0, 12: 0, 123: 0}

# Maximum number of frames to keep a marker's last detected position before considering it lost.
max_timeout = 15

# List to store the total traveled distance of the handlebar over time.
# The first entry is 0.0, indicating the starting point.
handlebar_traveled_distance = [0.0]

# Variable to store the last valid position of the handlebar for distance calculations.
last_handlebar_position = None

# Sampling rate for data processing, specified in frames per second (fps).
# This is used for timing and to scale data over time.
fps = 30


# ==========================
# Functions for Angle Calculation
# ==========================
def calculate_angles(marker_positions):
    """
    Calculates the knee and femur angles using the positions of three markers:
    hip (ID: 1), knee (ID: 12), and ankle (ID: 123). 

    Args:
        marker_positions (dict): Dictionary containing marker IDs as keys and their 
                                 (x, y) positions as values.

    Returns:
        tuple: A tuple containing:
               - femur_angle_avg (float): The average femur angle from the hip to the knee.
               - knee_angle_avg (float): The average knee angle between the femur and tibia.
               If the required markers are not detected, returns (None, None).
    """
    # Check if required markers are available
    if 1 in marker_positions and 12 in marker_positions and 123 in marker_positions:
        # Extract positions of the markers
        hip_pos = marker_positions[1]
        knee_pos = marker_positions[12]
        ankle_pos = marker_positions[123]

        # Calculate vectors between markers
        vec_hip_knee = np.array(knee_pos) - np.array(hip_pos)  # Vector from hip to knee
        vec_knee_ankle = np.array(ankle_pos) - np.array(knee_pos)  # Vector from knee to ankle

        # Compute the angle at the knee using the dot product formula
        cos_theta = np.dot(vec_hip_knee, vec_knee_ankle) / (np.linalg.norm(vec_hip_knee) * np.linalg.norm(vec_knee_ankle))
        knee_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip to avoid numerical issues

        # Compute the femur angle (relative to horizontal axis, adjusted to be in range 0-180 degrees)
        femur_angle = np.degrees(np.arctan2(vec_hip_knee[1], vec_hip_knee[0])) - 90
        if femur_angle < 0:  # Ensure femur angle is non-negative
            femur_angle = -femur_angle
        femur_angle = np.clip(femur_angle, 0, 180)

        # Update history deques for moving average calculations
        femur_angle_history.append(femur_angle)
        knee_angle_history.append(knee_angle)

        # Calculate moving averages
        femur_angle_avg = np.mean(femur_angle_history)
        knee_angle_avg = np.mean(knee_angle_history)

        # Return the calculated averages
        return femur_angle_avg, knee_angle_avg

    # If required markers are not detected, return None for both angles
    return None, None


# ==========================
# Functions for Marker Detection
# ==========================
def find_markers(frame):
    """
    Detects ArUco markers in the given frame and calculates the traveled distance
    for a specific marker (Marker ID: 2) simulating handlebar movement.

    Args:
        frame (numpy.ndarray): The current frame from the video feed (BGR format).

    Returns:
        tuple:
            - marker_positions (dict): A dictionary with marker IDs as keys and their
                                       (x, y) positions as values.
            - frame (numpy.ndarray): The frame with detected markers drawn on it.
    """
    global handlebar_traveled_distance, last_handlebar_position

    # Convert the frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the grayscale frame
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Dictionary to store detected marker positions
    marker_positions = {}

    # Process detected markers if any are found
    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]  # Extract marker ID
            if marker_id in [1, 12, 123, 2]:  # Filter for relevant markers
                # Calculate the marker's center position as the average of its corners
                position = np.mean(corner[0], axis=0)
                marker_positions[marker_id] = position

        # Draw detected markers on the frame for visualization
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Handle handlebar movement tracking for Marker 2
    if 2 in marker_positions:
        # Get the current position of Marker 2
        current_handlebar_position = marker_positions[2]

        if last_handlebar_position is not None:
            # Calculate vertical movement (change in y-axis position)
            vertical_distance = current_handlebar_position[1] - last_handlebar_position[1]

            # Estimate a conversion factor for pixels to centimeters
            cm_per_pixel = (
                8
                / np.linalg.norm(
                    corners[ids.tolist().index([2])][0][0]
                    - corners[ids.tolist().index([2])][0][1]
                )
            )

            # Convert the traveled vertical distance from pixels to centimeters
            traveled_distance = vertical_distance * cm_per_pixel

            # Update the handlebar traveled distance
            # Simulate "up and down" movement and ensure non-negative total distance
            new_distance = handlebar_traveled_distance[-1] + traveled_distance
            handlebar_traveled_distance.append(max(0, new_distance))

        # Update the last known position of the handlebar
        last_handlebar_position = current_handlebar_position

    return marker_positions, frame



# ==========================
# Functions for Measurement Loop
# ==========================
def measurement_loop():
    """
    Main measurement loop that processes video frames for real-time squat analysis.
    Detects markers, calculates angles, tracks squat repetitions, and updates the UI.

    Globals:
        measurement_active (bool): Controls whether the measurement loop is active.
        data_storage (dict): Stores femur angles, knee angles, and timestamps for analysis.
        squat_count (int): Tracks the number of completed squat repetitions.
        sound_enabled (bool): Toggles sound feedback when a squat is counted.
        rep_state (str): Tracks the current squat state ("up" or "down").

    Steps:
        1. Captures a video frame and processes it.
        2. Detects marker positions and calculates angles.
        3. Updates squat count based on angle thresholds and state transitions.
        4. Updates data storage for analysis and visualization.
        5. Refreshes the UI with the processed frame.

    Returns:
        None
    """
    global measurement_active, data_storage, squat_count, sound_enabled, rep_state

    # Exit the loop if measurement is not active
    if not measurement_active:
        return

    # Capture a frame from the video feed
    ret, frame = cap.read()
    if ret:
        # Detect markers and calculate angles
        marker_positions, frame = find_markers(frame)
        femur_angle, knee_angle = calculate_angles(marker_positions)
        update_marker_status(marker_positions)

        if femur_angle is not None:
            # Store calculated angles and timestamps for analysis
            current_time = time.time()
            data_storage['femur_angle'].append(femur_angle)
            data_storage['knee_angle'].append(knee_angle)
            data_storage['timestamps'].append(current_time)

            # Debugging: Output current values for monitoring
            print(f"Femur Angle: {femur_angle}, Knee Angle: {knee_angle}, Rep State: {rep_state}, Squat Count: {squat_count}")

            # Squat counting logic based on femur angle thresholds
            if femur_angle <= 90:  # When the thigh is parallel or lower than horizontal
                if rep_state == "up":  # If previously in "up" position
                    rep_state = "down"  # Transition to "down" state
                    print("Switch to down state")
            elif femur_angle > 90:  # When the thigh returns above the horizontal
                if rep_state == "down":  # If previously in "down" position
                    squat_count += 1  # Increment the squat count
                    rep_state = "up"  # Transition to "up" state
                    update_squat_count_label()  # Update the UI with the new count
                    print(f"Squat counted! New count: {squat_count}")
                    if sound_enabled:
                        winsound.Beep(1000, 200)  # Play a sound for feedback

            # Update visualization with squat analysis results
            update_visualization()

        # Convert the frame to a format compatible with Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = Image.fromarray(img)  # Convert to a PIL Image
        imgtk = ImageTk.PhotoImage(image=img)  # Convert to ImageTk format for Tkinter
        camera_label.imgtk = imgtk  # Keep a reference to prevent garbage collection
        camera_label.config(image=imgtk)  # Update the camera label with the new frame

    # Schedule the next iteration of the loop
    root.after(10, measurement_loop)


# ==========================
# Functions for GUI Updates
# ==========================
def update_marker_status(marker_positions):
    """
    Updates the GUI with the detection status of individual markers and overall status.

    Args:
        marker_positions (dict): Dictionary of detected marker IDs and their positions.

    Updates:
        - Hip status label: Indicates whether the hip marker (ID: 1) is detected.
        - Knee status label: Indicates whether the knee marker (ID: 12) is detected.
        - Ankle status label: Indicates whether the ankle marker (ID: 123) is detected.
        - Handlebar status label: Indicates whether the handlebar marker (ID: 2) is detected.
        - All markers status label: Indicates whether all required markers are detected.
    """
    hip_status.config(text="Hip: Detected" if 1 in marker_positions else "Hip: Not Detected")
    knee_status.config(text="Knee: Detected" if 12 in marker_positions else "Knee: Not Detected")
    ankle_status.config(text="Ankle: Detected" if 123 in marker_positions else "Ankle: Not Detected")
    handlebar_status.config(text="Handlebar: Detected" if 2 in marker_positions else "Handlebar: Not Detected")

    if 1 in marker_positions and 12 in marker_positions and 123 in marker_positions and 2 in marker_positions:
        all_markers_status.config(text="All Markers Detected!", foreground="green")
    else:
        all_markers_status.config(text="Waiting for Markers...", foreground="red")


def update_squat_count_label():
    """
    Updates the squat count label in the GUI.

    Updates:
        - Squat count label: Displays the current number of counted squats.
    """
    squat_count_label.config(text=f"Squat Count: {squat_count}")


def update_visualization():
    """
    Updates the visualization components in the GUI, including:
    - The knee angle plot.
    - The handlebar travel plot.
    - Current femur angle and knee angle labels.
    - Handlebar travel distance label.

    Logic:
        - Displays data from the last 5 seconds, based on the frames per second (fps).
        - Updates plots for knee angle and handlebar travel.
        - Redraws the canvas for updated visuals.
    """
    # Calculate the number of data points to display for the last 5 seconds
    points_to_display = 5 * fps

    # Update Knee Angle plot with the last 5 seconds of data
    knee_angle_line.set_ydata(data_storage['knee_angle'][-points_to_display:])
    knee_angle_line.set_xdata(range(len(data_storage['knee_angle'][-points_to_display:])))

    ax2.relim()
    ax2.autoscale_view()

    # Update Handlebar Distance plot
    ax3.clear()
    ax3.set_title("Handlebar Travel")
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("Distance (cm)")
    ax3.plot(handlebar_traveled_distance[-points_to_display:], label="Handlebar Travel", color="green")
    ax3.legend()

    # Redraw the canvas with updated data
    canvas.draw()

    # Update angle labels
    if data_storage['femur_angle']:
        femur_angle_label.config(text=f"Femur Angle: {data_storage['femur_angle'][-1]:.2f}°")
    if data_storage['knee_angle']:
        knee_angle_label.config(text=f"Knee Angle: {data_storage['knee_angle'][-1]:.2f}°")

    # Update handlebar distance label
    if handlebar_traveled_distance:
        handlebar_distance_label.config(text=f"Handlebar Travel: {handlebar_traveled_distance[-1]:.2f} cm")


# ==========================
# Functions for GUI Controls
# ==========================
def start_measurement():
    """
    Starts the squat measurement process:
    - Activates the measurement loop.
    - Opens a connection to the camera for real-time video capture.

    Globals:
        measurement_active (bool): Set to True to enable measurement.
        cap (cv2.VideoCapture): Video capture object initialized for camera input.

    Logic:
        - Opens the default camera (index 0) using OpenCV.
        - If the camera cannot be accessed, prints an error message.
        - Starts the `measurement_loop` for real-time processing.
    """
    global measurement_active, cap
    measurement_active = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the default camera
    if not cap.isOpened():  # Check if the camera opened successfully
        print("Cannot open camera")
        return
    measurement_loop()  # Start the main measurement loop


def stop_measurement():
    """
    Stops the squat measurement process:
    - Deactivates the measurement loop.
    - Releases the camera and closes any OpenCV windows.
    - Resets the camera display in the GUI.

    Globals:
        measurement_active (bool): Set to False to stop measurement.
        cap (cv2.VideoCapture): Releases the video capture object if it exists.

    Logic:
        - Sets `measurement_active` to False to terminate the loop.
        - Releases the camera resource using `cap.release()`.
        - Clears the `camera_label` in the GUI to display an empty screen.
    """
    global measurement_active, cap
    measurement_active = False
    if cap:
        cap.release()  # Release the camera resource
        cv2.destroyAllWindows()  # Close any OpenCV windows

    # Clear the camera_label in the GUI
    camera_label.config(image='')  # Resets the GUI to show no video


def reset_counter():
    """
    Resets the squat count to zero and updates the GUI.

    Globals:
        squat_count (int): Resets the global squat count variable to 0.

    Updates:
        - Calls `update_squat_count_label()` to refresh the GUI squat count label.
    """
    global squat_count
    squat_count = 0  # Reset the squat count
    update_squat_count_label()  # Update the GUI label


def toggle_sound():
    """
    Toggles the sound feedback for squat counting.

    Globals:
        sound_enabled (bool): Toggles the global flag to enable or disable sound.

    Logic:
        - When `sound_enabled` is True, plays a beep sound for feedback after each squat.
        - When `sound_enabled` is False, disables the sound feedback.
    """
    global sound_enabled
    sound_enabled = not sound_enabled  # Toggle the sound feedback flag


# ==========================
# GUI Initialization
# ==========================
"""
This section initializes the GUI for the squat measurement application.
It includes controls for starting/stopping the measurement, displaying feedback
on detected markers, and visualizing real-time squat analysis data.

The GUI layout consists of:
1. Control Buttons (Start, Stop, Reset, Enable Sound).
2. Status Labels (Marker Detection, Squat Count, Angles, Handlebar Distance).
3. Camera Feed Display.
4. Matplotlib Plots for Real-Time Data Visualization.
"""

# Create the main application window
root = tk.Tk()
root.title("Squat Measurement GUI")

# Configure the root window to dynamically resize widgets
"""
- Makes the application responsive by allowing rows and columns to expand with window resizing.
- Grid system ensures consistent alignment of elements.
"""
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Create a frame to contain all widgets
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure the frame for dynamic resizing
"""
- Adds weights to rows and columns within the frame.
- Ensures components like buttons and labels resize proportionally.
"""
for i in range(13):  # Configuring rows for 13 UI components
    frame.rowconfigure(i, weight=1)
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

# Add Control Buttons
"""
- Start Button: Activates the measurement loop and camera feed.
- Stop Button: Deactivates the measurement loop and releases the camera.
- Reset Button: Resets the squat count to 0.
"""
start_button = ttk.Button(frame, text="Start", command=start_measurement)
start_button.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

stop_button = ttk.Button(frame, text="Stop", command=stop_measurement)
stop_button.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

reset_button = ttk.Button(frame, text="Reset Counter", command=reset_counter)
reset_button.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Squat Count Display
"""
- Displays the current number of squat repetitions.
- Updated dynamically during the measurement loop.
"""
squat_count_label = ttk.Label(frame, text=f"Squat Count: {squat_count}")
squat_count_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

# Sound Toggle
"""
- Checkbutton to enable or disable sound feedback for squat counting.
"""
sound_checkbox = ttk.Checkbutton(frame, text="Enable Sound", command=toggle_sound)
sound_checkbox.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))

# Marker Detection Status Labels
"""
- Display detection status for hip, knee, ankle, and handlebar markers.
- Updated dynamically during measurement.
"""
hip_status = ttk.Label(frame, text="Hip: Not Detected")
hip_status.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

knee_status = ttk.Label(frame, text="Knee: Not Detected")
knee_status.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

ankle_status = ttk.Label(frame, text="Ankle: Not Detected")
ankle_status.grid(row=6, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

handlebar_status = ttk.Label(frame, text="Handlebar: Not Detected")
handlebar_status.grid(row=7, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

all_markers_status = ttk.Label(frame, text="Waiting for Markers", foreground="red")
all_markers_status.grid(row=8, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Angle and Handlebar Distance Labels
"""
- Display the most recent femur and knee angles, as well as the handlebar distance traveled.
"""
femur_angle_label = ttk.Label(frame, text="Femur Angle: N/A")
femur_angle_label.grid(row=9, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

knee_angle_label = ttk.Label(frame, text="Knee Angle: N/A")
knee_angle_label.grid(row=10, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

handlebar_distance_label = ttk.Label(frame, text="Handlebar Distance: N/A")
handlebar_distance_label.grid(row=11, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Camera Feed Display
"""
- Displays the live camera feed processed during the measurement loop.
"""
camera_label = tk.Label(root)
camera_label.grid(row=11, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

# Matplotlib Visualization
"""
- Two subplots for visualizing:
    1. Knee Angle over Time
    2. Handlebar Distance over Time
- Configured for clear labels, titles, and scaling.
"""
fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(5, 8))
fig.subplots_adjust(hspace=0.5)

# Configure the second subplot for knee angle
ax2.set_ylim(0, 180)  # Angle range in degrees
ax2.set_xlim(0, 150)  # Number of frames
ax2.set_ylabel('Knee Angle (degrees)')
ax2.set_xlabel('Time (frames)')
knee_angle_line, = ax2.plot([], [], label='Knee Angle', color='red')
ax2.legend()

# Configure the third subplot for handlebar distance
ax3.set_ylim(0, 100)  # Expected handlebar travel distance
ax3.set_xlim(0, 150)  # Number of frames
ax3.set_ylabel('Distance (cm)')
ax3.set_xlabel('Time (frames)')
ax3.set_title('Handlebar Distance Over Time')

# Embed Matplotlib Figure in Tkinter
"""
- Integrates the Matplotlib figure into the Tkinter GUI.
- Dynamically updated during the measurement loop.
"""
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure dynamic resizing for rows and columns
"""
- Ensures the camera feed and plots resize proportionally when the window is resized.
"""
root.rowconfigure(11, weight=1)  # Camera frame row
root.rowconfigure(12, weight=1)  # Plot row
root.columnconfigure(0, weight=1)

# Start the Tkinter main event loop
"""
- Keeps the application running and responsive to user interactions.
"""
root.mainloop()

