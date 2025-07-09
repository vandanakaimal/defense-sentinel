# defense_sentinel_app.py
import streamlit as st
import cv2
from ultralytics import YOLO
import datetime
import os
from PIL import Image

# Setup
st.set_page_config(page_title="Defense Sentinel", layout="centered")
st.title("🛡️ Defense Sentinel: AI Intrusion Detection")

# Output directory
output_dir = "intrusions"
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Make sure this is downloaded

# Toggle webcam
run_detection = st.toggle("Enable Camera Feed")
placeholder = st.empty()

# Log section
st.subheader("📜 Intrusion Logs")
log_box = st.empty()
log_messages = []

if run_detection:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        results = model(frame)
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                img_path = f"{output_dir}/intrusion_{timestamp}.jpg"
                cv2.imwrite(img_path, frame)
                log_messages.insert(0, f"⚠️ Intrusion Detected: {label} at {timestamp}")

        # Annotate and show
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        placeholder.image(annotated_frame_rgb, channels="RGB")
        log_box.markdown("\n".join(log_messages[:10]))

        # Stop if toggle is off
        if not st.session_state.get("run_detection", True):
            break

    cap.release()
else:
    st.info("Toggle the switch above to start intrusion detection.")

# Show saved intrusions
with st.expander("📁 View Saved Intrusion Snapshots"):
    images = os.listdir(output_dir)
    images.sort(reverse=True)
    for img in images[:5]:
        st.image(Image.open(os.path.join(output_dir, img)), caption=img)

st.caption("Built for CAIR-DRDO Portfolio | by Vandana")
