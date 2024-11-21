# Squat Measurement Application

## Overview
This Python application tracks squat performance using ArUco markers, analyzing femur and knee angles via a live camera feed.

---

## Requirements

### Software
1. **Python 3.7+**
2. Required libraries (install via pip):  
   ```bash
   pip install opencv-python numpy matplotlib Pillow
   ```
3. **OpenCV**: Ensure `cv2.aruco` is available.

### Hardware
- **Camera**: Standard webcam at 30 FPS.
- **Environment**: Bright lighting, white wall background, stable camera placement.

---

## Marker Setup

1. **Markers**: Use `DICT_4X4_250`, sized **8x8 cm** with a white border.
2. **Positions**: Attach markers securely:
   - **Hip (ID: 1)**: Side of pelvis.
   - **Knee (ID: 12)**: Side of knee.
   - **Ankle (ID: 123)**: Side of ankle.
   - **Handlebar (ID: 2)**: Center of handlebar.
3. **Clothing**: Wear fitted, non-reflective clothing.

---

## Camera and Environment

- **Camera Placement**: Height of **65-75 cm**, 2 meters from the person, capturing all markers in the frame.
- **Positioning**: User should face **side-view** to the camera.

---

## Usage

1. Run the script:
   ```bash
   python squat_measurement.py
   ```
2. GUI Controls:
   - **Start**: Begin squat tracking.
   - **Stop**: Stop tracking.
   - **Reset Counter**: Reset squat count.
   - **Enable Sound**: Toggle beep feedback.

---

## Best Practices

- Ensure all markers are **visible** and stay within the camera frame.
- Perform squats fully (femur angle < 90° for a valid count).
- Use a **simple background** for best detection accuracy.

---

## Troubleshooting

- **Markers Not Detected**: Check visibility, lighting, and dimensions.
- **Squats Not Counted**: Ensure full squat range and proper marker placement.