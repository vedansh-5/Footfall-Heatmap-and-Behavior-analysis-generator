import streamlit as st
import cv2
import numpy as np

from detector import run_detection
from tracker import track_video
from bev_model import MonoLayoutBEV
from bev_floor_plan import generate_floor_plan
from bev_mapper import map_tracks_to_bev
from generate_bev_heatmap import generate_bev_heatmap
from homography_config import src_points as default_src_points # Rename default
from plan_heatmap import generate_plan_heatmap_from_csv, parse_points_str
from plan_point_picker import pick_plan_points
from plan_heatmap_service import run_and_save_plan_heatmap


import os
import pandas as pd

st.title("Footfall HeatMap & Behaviour Analysis")

uploaded_video = st.file_uploader("Upload CCTV Video", type=["mp4", "mov", "avi"])
csv_path = st.text_input("Tracking CSV Path", value="output/tracks.csv")



if uploaded_video:
    video_path = os.path.join("input_videos", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

    if st.button("Run Detection"):
        os.makedirs("output/detections", exist_ok=True)
        output_path = "output/detections/detected.mp4"
        run_detection(video_path, output_path)
        st.success("Detection complete!")
        st.video(output_path)
    
    if st.button("Run Tracking"):
        os.makedirs("output/tracks", exist_ok=True)
        output_track_video="output/tracks/tracked.mp4"
        tracked_video, csv_path = track_video(video_path, output_track_video)
        st.success("Tracking complete!")
        st.video(tracked_video)
        st.write(f"CSV saved at {csv_path}")



    
st.markdown("---")
st.header("Heatmap on Floor Plan")

st.subheader("1. Define Source Points (Video)")
st.info("These are the 4 corner points from the **video frame** that correspond to the plan area.")
default_src_str = "; ".join([f"{p[0]},{p[1]}" for p in default_src_points])
src_points_str = st.text_area("Source Points (x1,y1; x2,y2; ...)", value=default_src_str, height=100)

st.subheader("2. Pick Destination Points (Plan)")
plan_file = st.file_uploader("Upload Floor Plan Image", type=["png", "jpg", "jpeg"], key="plan_upload")
plan_path, dst_pts = pick_plan_points(plan_file, state_key="plan_points")

st.markdown("---")
st.subheader("3. Generate Heatmap")

kernel_size = st.slider("Heatmap smoothing (kernel size)", min_value=5, max_value=201, value=75, step=2)
sigma = st.slider("Gaussian sigma (0 = auto)", min_value=0, max_value=100, value=0, step=1)
alpha = st.slider("Overlay opacity", min_value=0.1, max_value=0.95, value=0.6, step=0.05)

if st.button("Generate Plan Heatmap"):
    src_points = None
    try:
        src_points = parse_points_str(src_points_str)
    except Exception as e:
        st.error(f"Invalid Source Points format: {e}")

    if src_points is None:
        pass 
    elif not plan_path:
        st.error("Please upload a floor plan image first.")
    elif not csv_path or not os.path.isfile(csv_path):
        st.error(f"Tracking CSV not found at '{csv_path}'. Please run tracking or verify the path.")
    elif dst_pts is None or len(dst_pts) != 4:
        st.error("Please select exactly 4 points on the plan before generating.")
    else:
        try:
            result = run_and_save_plan_heatmap(
                csv_path=csv_path,
                plan_path=plan_path,
                src_points=src_points,
                dst_pts=dst_pts,
                kernel_size=kernel_size,
                sigma=sigma,
                alpha=alpha,
            )
            st.success("Plan heatmap generated.")
            st.image(cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB), caption="Heatmap Overlay", use_container_width=True)
            st.write("Saved files:")
            st.code(os.path.abspath(result["paths"]["overlay"]))
            st.code(os.path.abspath(result["paths"]["heat_u8"]))
            st.code(os.path.abspath(result["paths"]["heat_color"]))
            st.download_button(
                "Download Overlay",
                data=cv2.imencode(".png", result["overlay_bgr"])[1].tobytes(),
                file_name="heatmap_overlay.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(str(e))