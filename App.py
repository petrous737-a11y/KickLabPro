import streamlit as st
import cv2
import tempfile
import math
import numpy as np
from ultralytics import YOLO

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Kick Lab V2 Pro", layout="wide")

# ======================
# FIXED LOGO (IMPORTANT)
# ======================
st.sidebar.image("https://i.imgur.com/your-logo.png", width=160)

st.sidebar.title("⚽ Kick Lab V2 Pro")
st.sidebar.write("AI Football Analysis System")

st.title("⚽ Kick Lab AI V2 - Professional Edition")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ======================
# UPLOAD VIDEO
# ======================
video = st.file_uploader("Upload Football Video", type=["mp4","mov","avi"])

# ======================
# SIMPLE xG MODEL
# ======================
def xg_model(speed):
    return min(1.0, speed / 1200)

# ======================
# PROCESS
# ======================
if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)

    points = []
    speeds = []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        frame_rate = 30

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # ball detection
                if cls == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    points.append((cx, cy))

                    cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)

        # draw trajectory
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (0,255,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

    # ======================
    # ANALYSIS
    # ======================
    st.subheader("📊 Kick Lab AI Analysis V2")

    if len(points) > 2:

        total_dist = 0

        for i in range(1, len(points)):
            x1,y1 = points[i-1]
            x2,y2 = points[i]

            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_dist += dist

            speed = dist * frame_rate
            speeds.append(speed)

        avg_speed = sum(speeds) / len(speeds)
        xg = xg_model(avg_speed)

        col1, col2, col3 = st.columns(3)

        col1.metric("Avg Speed", f"{avg_speed:.2f}")
        col2.metric("xG Estimate", f"{xg:.2f}")
        col3.metric("Tracking Points", len(points))

        st.success("AI Analysis Completed ⚽")

        # Heatmap
        heatmap = np.zeros((100,100))

        for (x,y) in points:
            hx = min(99, abs(x)//10)
            hy = min(99, abs(y)//10)
            heatmap[hy][hx] += 1

        st.subheader("🔥 Heatmap")
        st.image(heatmap, clamp=True)
