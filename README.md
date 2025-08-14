# Live Face Recognition on Streamlit (WebRTC + OpenCV LBPH)

This app lets you enroll faces, train an LBPH model (OpenCV), and run **LIVE** face recognition in the browser using your webcam.

## Features
- Enroll via webcam capture or image upload
- Train LBPH (no heavy dlib/torch)
- Live video via `streamlit-webrtc` with on-frame recognition
- Works on Streamlit Community Cloud (free)

## How to Run Locally
```bash
python -m venv .venv
# On Linux/Mac
source .venv/bin/activate
# On Windows
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)
1. Push these files to a **public GitHub repo** (3 files: `app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io/ and **New app** → select your repo/branch → set **Main file path** to `app.py` → Deploy.
3. After it builds, open the app URL.

## Usage
1. **Enroll** tab: Enter a name, capture images or upload a few for each person.
2. **Train** tab: Click **Train / Retrain**.
3. **Live Recognition** tab: Click **Start** and allow camera. You’ll see boxes and names in real-time.

### Tips for Accuracy
- Add 10–20 varied images per person.
- Use good frontal faces; avoid strong backlight.
- Tune the “Recognition threshold” slider. Lower numbers = more strict match. Start ~60 and adjust.

## Notes
- The LBPH algorithm returns a **lower value for better match**. Threshold typical range: 40–90 depending on your data.
- If you later move to Raspberry Pi, you can reuse the **data/** and **models/** folders. Install `opencv-contrib-python` and run a similar script to load `.xml` model and `labels.json`.

## Troubleshooting
- On Streamlit Cloud, always use `opencv-contrib-python-headless`. Do not install both `opencv-python` and `opencv-contrib-python` together.
- If your camera doesn't appear in mobile browsers, ensure HTTPS and grant camera permissions.
