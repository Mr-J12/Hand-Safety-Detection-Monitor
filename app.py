import streamlit as st
import cv2
import numpy as np
import math
import time
import tempfile
import os


def get_hand_contour(frame, lower_skin, upper_skin):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            return max_contour
    return None


def get_fingertip(contour):
    extTop = tuple(contour[contour[:, :, 1].argmin()][0])
    return extTop


st.set_page_config(page_title="Hand Safety Detection Monitor", layout="wide")
st.title("Hand Safety Detection Monitor")

col1, col2 = st.columns([3, 1])

with col2:
    st.header("Controls")
    run = st.checkbox("Run Camera", value=False, key="run_camera")
    source = st.radio("Input Source", ["Local Webcam (local only)", "Upload video/image", "Browser camera (webrtc)"], index=0, key="source_mode")
    st.markdown("---")
    obj_x = st.slider("Object Center X", 0, 1280, 150)
    obj_y = st.slider("Object Center Y", 0, 960, 240)
    obj_radius = st.slider("Object Radius", 10, 300, 50)
    dist_warning = st.slider("Warning Distance (px)", 0, 500, 150)
    dist_danger = st.slider("Danger Distance (px)", 0, 200, 20)
    st.markdown("---")
    st.markdown("If the app cannot access a webcam, make sure your browser/OS allows camera access and that no other app (e.g., another camera app) is using the webcam.")

with col1:
    frame_window = st.image(np.zeros((480, 640, 3), dtype=np.uint8))
    status_text = st.empty()

uploaded_file = None
if source == "Upload video/image":
    uploaded_file = st.file_uploader("Upload image or video", type=["png", "jpg", "jpeg", "mp4", "mov", "avi"], key="uploader")
elif source == "Browser camera (webrtc)":
    st.info("Browser camera mode requires `streamlit-webrtc`. Install it and re-run, or use 'Local Webcam' when running locally.")

# HSV skin range defaults (can be adjusted in code if needed)
lower_skin = np.array([0, 10, 162], dtype=np.uint8)
upper_skin = np.array([26, 110, 255], dtype=np.uint8)

cap = None
prev_frame_time = 0

try:
    if run:
        obj_center = (obj_x, obj_y)
        # Local webcam mode
        if source == "Local Webcam (local only)":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to open webcam. Check camera permissions or device." )
            else:
                st.success("Camera started")
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        status_text.error("Failed reading from camera")
                        break

                    frame = cv2.flip(frame, 1)
                    # process frame (same as before)
                    hand_contour = get_hand_contour(frame, lower_skin, upper_skin)
                    state_color = (0, 255, 0)
                    state_text = "SAFE"
                    if hand_contour is not None:
                        cv2.drawContours(frame, [hand_contour], -1, (255, 255, 0), 2)
                        fingertip = get_fingertip(hand_contour)
                        fx, fy = fingertip
                        cv2.circle(frame, fingertip, 8, (255, 0, 255), -1)
                        pixel_dist = math.sqrt((obj_x - fx)**2 + (obj_y - fy)**2)
                        distance_to_edge = pixel_dist - obj_radius
                        if distance_to_edge <= dist_danger:
                            state_text = "DANGER DANGER"
                            state_color = (0, 0, 255)
                        elif distance_to_edge <= dist_warning:
                            state_text = "WARNING"
                            state_color = (0, 165, 255)
                        cv2.line(frame, fingertip, (obj_x, obj_y), state_color, 2)

                    thickness = 3 if state_text != "DANGER DANGER" else -1
                    cv2.circle(frame, (obj_x, obj_y), obj_radius, state_color, thickness)
                    cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
                    cv2.putText(frame, f"State: {state_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 3)
                    if state_text == "DANGER DANGER":
                        cv2.putText(frame, "DANGER DANGER", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    new_frame_time = time.time()
                    if prev_frame_time != 0:
                        fps = 1 / (new_frame_time - prev_frame_time)
                    else:
                        fps = 0
                    prev_frame_time = new_frame_time
                    cv2.putText(frame, f"FPS: {int(fps)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels='RGB')
                    time.sleep(0.01)

        # Upload mode: accept image or video and process
        elif source == "Upload video/image":
            if uploaded_file is None:
                st.info("Upload an image or video file to start processing.")
            else:
                # determine type
                content = uploaded_file.read()
                # image
                if uploaded_file.type.startswith("image"):
                    npimg = np.frombuffer(content, np.uint8)
                    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                    if frame is None:
                        st.error("Could not decode image")
                    else:
                        frame = cv2.flip(frame, 1)
                        hand_contour = get_hand_contour(frame, lower_skin, upper_skin)
                        state_color = (0, 255, 0)
                        state_text = "SAFE"
                        if hand_contour is not None:
                            cv2.drawContours(frame, [hand_contour], -1, (255, 255, 0), 2)
                            fingertip = get_fingertip(hand_contour)
                            fx, fy = fingertip
                            cv2.circle(frame, fingertip, 8, (255, 0, 255), -1)
                            pixel_dist = math.sqrt((obj_x - fx)**2 + (obj_y - fy)**2)
                            distance_to_edge = pixel_dist - obj_radius
                            if distance_to_edge <= dist_danger:
                                state_text = "DANGER DANGER"
                                state_color = (0, 0, 255)
                            elif distance_to_edge <= dist_warning:
                                state_text = "WARNING"
                                state_color = (0, 165, 255)
                            cv2.line(frame, fingertip, (obj_x, obj_y), state_color, 2)
                        thickness = 3 if state_text != "DANGER DANGER" else -1
                        cv2.circle(frame, (obj_x, obj_y), obj_radius, state_color, thickness)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_window.image(frame_rgb, channels='RGB')

                # video
                else:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.type.split('/')[-1])
                    try:
                        tmp.write(content)
                        tmp.flush()
                        tmp.close()
                        cap = cv2.VideoCapture(tmp.name)
                        if not cap.isOpened():
                            st.error("Unable to open uploaded video")
                        else:
                            st.success("Processing uploaded video")
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame = cv2.flip(frame, 1)
                                hand_contour = get_hand_contour(frame, lower_skin, upper_skin)
                                state_color = (0, 255, 0)
                                state_text = "SAFE"
                                if hand_contour is not None:
                                    cv2.drawContours(frame, [hand_contour], -1, (255, 255, 0), 2)
                                    fingertip = get_fingertip(hand_contour)
                                    fx, fy = fingertip
                                    cv2.circle(frame, fingertip, 8, (255, 0, 255), -1)
                                    pixel_dist = math.sqrt((obj_x - fx)**2 + (obj_y - fy)**2)
                                    distance_to_edge = pixel_dist - obj_radius
                                    if distance_to_edge <= dist_danger:
                                        state_text = "DANGER DANGER"
                                        state_color = (0, 0, 255)
                                    elif distance_to_edge <= dist_warning:
                                        state_text = "WARNING"
                                        state_color = (0, 165, 255)
                                    cv2.line(frame, fingertip, (obj_x, obj_y), state_color, 2)
                                thickness = 3 if state_text != "DANGER DANGER" else -1
                                cv2.circle(frame, (obj_x, obj_y), obj_radius, state_color, thickness)
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_window.image(frame_rgb, channels='RGB')
                                time.sleep(0.01)
                            cap.release()
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass

finally:
    if cap is not None and cap.isOpened():
        cap.release()
