# Hand Safety Detection Monitor

This repository contains a Streamlit app to run the Hand Safety Monitor using your webcam.

Files added:
- `app.py` — Streamlit application that captures webcam frames, detects the hand, computes proximity to a virtual object, and displays safety state.
- `requirements.txt` — Python dependencies.

How to run locally

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL that Streamlit prints (usually `http://localhost:8501`).

Notes
- Ensure no other application is using the webcam.
- If you have issues with `opencv-python` on your platform, consider installing the correct distribution for your OS.

Deployment notes
- If you deploy this app to Streamlit Cloud or other server environments, replace `opencv-python` with `opencv-python-headless` (already done in `requirements.txt`) because server hosts often lack GUI libraries required by the regular OpenCV wheel. This avoids import-time native errors when importing `cv2`.
- Streamlit Cloud (and most server hosts) cannot access your local machine's webcam. The app attempts to use `cv2.VideoCapture(0)` which only works when running locally where a physical webcam is present. For cloud deployment consider one of the following:
	- Run the app locally instead of on cloud (recommended for webcam use).
	- Modify the app to accept uploaded video files or images from users.
	- Use a browser-based camera stream component such as `streamlit-webrtc` or community camera widgets (these run in the browser and can forward frames to the server).

