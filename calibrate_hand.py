import cv2
import numpy as np

# Global variables
hsv_lower = np.array([0, 0, 0])
hsv_upper = np.array([179, 255, 255])
calibrated = False

def pick_color(event, x, y, flags, param):
    global hsv_lower, hsv_upper, calibrated
    
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        
        # 1. Grab a small 5x5 ROI (Region of Interest) around the click
        roi = frame[y-2:y+3, x-2:x+3]
        
        if roi.size == 0: return

        # 2. Convert that small ROI to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 3. Calculate the average color
        mean_hsv = np.mean(hsv_roi, axis=(0, 1))
        
        # 4. Define offsets (Thresholds)
        # We allow a small range for Hue, but larger for Saturation/Value
        # to account for shadows and highlights.
        hue_buffer = 15
        sat_buffer = 50
        val_buffer = 50
        
        # 5. Calculate Lower and Upper Bounds
        # Note: We clip values so they don't go outside valid ranges
        # H: 0-179, S: 0-255, V: 0-255
        h_low = max(0, mean_hsv[0] - hue_buffer)
        s_low = max(0, mean_hsv[1] - sat_buffer)
        v_low = max(0, mean_hsv[2] - val_buffer)
        
        h_high = min(179, mean_hsv[0] + hue_buffer)
        s_high = min(255, mean_hsv[1] + sat_buffer)
        v_high = min(255, mean_hsv[2] + val_buffer)
        
        hsv_lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        hsv_upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        
        calibrated = True
        
        # Print the values to the terminal so the user can copy them
        print("\n=== CALIBRATION SUCCESSFUL ===")
        print(f"lower_skin = np.array([{int(h_low)}, {int(s_low)}, {int(v_low)}], dtype=np.uint8)")
        print(f"upper_skin = np.array([{int(h_high)}, {int(s_high)}, {int(v_high)}], dtype=np.uint8)")
        print("==============================\n")

def main():
    global calibrated
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow("Calibration")
    print("Click on your hand to calibrate. Press 'q' to quit, 'r' to reset.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # Set the mouse callback (passing the current frame as param)
        cv2.setMouseCallback("Calibration", pick_color, frame)
        
        # If calibrated, show the segmented view (Mask)
        if calibrated:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            
            # Combine mask with original frame for a "highlighted" look
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Display side-by-side: Original vs Result
            # Resize for cleaner display if needed
            display = np.hstack((frame, result))
            
            cv2.putText(display, "Calibrated! Check Terminal for values.", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display = frame
            cv2.putText(display, "Click on your hand to calibrate", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            calibrated = False
            print("Calibration reset.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()