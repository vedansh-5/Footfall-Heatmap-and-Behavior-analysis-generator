import streamlit as st
import cv2

from detector import run_detection
from tracker import track_video
from heatmap import generate_heatmap
from bev_model import MonoLayoutBEV
from bev_floor_plan import generate_floor_plan
from bev_mapper import map_tracks_to_bev
from generate_bev_heatmap import generate_bev_heatmap

import os
import pandas as pd

st.title("Footfall HeatMap & Behaviour Analysis")

uploaded_video = st.file_uploader("Upload CCTV Video", type=["mp4", "mov", "avi"])
csv_path = st.text_input("Tracking CSV Path")

if uploaded_video:
    video_path = os.path.join("input_videos", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

    if st.button("Run Detection"):
        output_path = "output/detections/detected.mp4"
        run_detection(video_path, output_path)
        st.success("Detection complete!")
        st.video(output_path)
    
    if st.button("Run Tracking"):
        output_track_video="output/tracks/tracked.mp4"
        tracked_video, csv_path = track_video(video_path, output_track_video)
        st.success("Tracking complete!")
        st.video(tracked_video)
        st.write(f"CSV saved at {csv_path}")
        
    if st.button("Generate BEV Heatmap"):
        bev_model = MonoLayoutBEV()
        cap = cv2.VideoCapture(uploaded_video.name)
        ret, frame = cap.read()
        cap.release()

        bev = bev_model.infer_bev(frame)
        plan_path = generate_floor_plan(bev)

        mapped = map_tracks_to_bev(csv_path)
        final_heatmap = generate_bev_heatmap(mapped, plan_path)

        st.image(final_heatmap)

