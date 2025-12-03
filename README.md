# Hand Safety Monitor — Streamlit Deployment

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
