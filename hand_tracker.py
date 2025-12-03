import cv2
import numpy as np
import math
import time

class HandSafetyMonitor:
    def __init__(self):
        # Initialize Camera
        self.cap = cv2.VideoCapture(0)
        
        # Virtual Object Properties (Center X, Center Y, Radius)
        self.obj_center = (150, 240) # Fixed position on screen
        self.obj_radius = 50
        
        # Distance Thresholds (in pixels)
        self.DIST_WARNING = 150
        self.DIST_DANGER = 20
        
        # Performance monitoring
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def get_hand_contour(self, frame):
        """
        Detects hand using HSV color segmentation.
        Note: You may need to tune 'lower' and 'upper' HSV bounds 
        depending on your lighting and skin tone.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Standard skin color range in HSV
        # Lower: [Hue, Saturation, Value]
        lower_skin = np.array([0, 10, 162], dtype=np.uint8)
        upper_skin = np.array([26, 110, 255], dtype=np.uint8)

        # Create binary mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Noise removal (Morphological operations)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Assume largest contour is the hand
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 1000: # Filter small noise
                return max_contour
        return None

    def get_fingertip(self, contour):
        """Finds the top-most point of the contour (fingertip approximation)."""
        # extTop is the point with the minimum Y value
        extTop = tuple(contour[contour[:, :, 1].argmin()][0])
        return extTop

    def run(self):
        print("Starting Safety Monitor. Press 'q' to exit.")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # 1. Processing
            hand_contour = self.get_hand_contour(frame)
            
            state_color = (0, 255, 0) # Default Green (SAFE)
            state_text = "SAFE"
            distance = 9999

            # 2. Virtual Object & Logic
            if hand_contour is not None:
                # Draw Hand Contour
                cv2.drawContours(frame, [hand_contour], -1, (255, 255, 0), 2)
                
                # Get Fingertip location
                fingertip = self.get_fingertip(hand_contour)
                fx, fy = fingertip
                
                # Draw Fingertip
                cv2.circle(frame, fingertip, 8, (255, 0, 255), -1)

                # Calculate Euclidean Distance to Object Center
                # dist = sqrt((x2-x1)^2 + (y2-y1)^2)
                pixel_dist = math.sqrt((self.obj_center[0] - fx)**2 + (self.obj_center[1] - fy)**2)
                
                # Distance to the EDGE of the object
                distance_to_edge = pixel_dist - self.obj_radius

                # 3. State Machine
                if distance_to_edge <= self.DIST_DANGER:
                    state_text = "DANGER DANGER"
                    state_color = (0, 0, 255) # Red
                elif distance_to_edge <= self.DIST_WARNING:
                    state_text = "WARNING"
                    state_color = (0, 165, 255) # Orange/Yellow (BGR)
                
                # Visual Line connecting finger to object
                cv2.line(frame, fingertip, self.obj_center, state_color, 2)

            # 4. Rendering Visuals
            
            # Draw Virtual Object (Circle)
            # Change circle thickness based on state (Fill it if danger)
            thickness = 3 if state_text != "DANGER DANGER" else -1
            cv2.circle(frame, self.obj_center, self.obj_radius, state_color, thickness)
            
            # Draw UI Overlay
            cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1) # Text Background
            cv2.putText(frame, f"State: {state_text}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 3)
            
            if state_text == "DANGER DANGER":
                # Flash effect or large center text
                cv2.putText(frame, "DANGER DANGER", (150, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            # FPS Calculation
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display
            cv2.imshow("Hand Safety Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandSafetyMonitor()
    app.run()