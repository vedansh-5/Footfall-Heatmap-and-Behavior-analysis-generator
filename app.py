import streamlit as st
import cv2
import numpy as np

from detector import run_detection
from tracker import track_video
from bev_model import MonoLayoutBEV
from bev_floor_plan import generate_floor_plan
from bev_mapper import map_tracks_to_bev
from generate_bev_heatmap import generate_bev_heatmap
from homography_config import src_points
from plan_heatmap import generate_plan_heatmap_from_csv, parse_points_str
from plan_point_picker import pick_plan_points             # NEW
from plan_heatmap_service import run_and_save_plan_heatmap # NEW


import os
import pandas as pd

st.title("Footfall HeatMap & Behaviour Analysis")

uploaded_video = st.file_uploader("Upload CCTV Video", type=["mp4", "mov", "avi"])
csv_path = st.text_input("Tracking CSV Path")

# Allow uploading BEV model weights and store to expected filename
# with st.expander("BEV model weights (MonoLayout) - optional", expanded=False):
#     st.write("Upload your MonoLayout .pt/.pth file. It will be saved as monolayout_pretrained.ot (expected by bev_model.py).")
#     weights_file = st.file_uploader("Upload weights", type=["pt", "pth"], key="monolayout_weights")
#     if weights_file is not None:
#         buggy_expected = "monolayout_pretrained.ot"  # bev_model.py expects this exact name
#         with open(buggy_expected, "wb") as f:
#             f.write(weights_file.getbuffer())
#         st.success(f"Weights saved to ./{buggy_expected}")

if uploaded_video:
    video_path = os.path.join("input_videos", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

    if st.button("Run Detection"):
        os.makedirs("output/detections", exist_ok=True)  # ensure folder
        output_path = "output/detections/detected.mp4"
        run_detection(video_path, output_path)
        st.success("Detection complete!")
        st.video(output_path)
    
    if st.button("Run Tracking"):
        os.makedirs("output/tracks", exist_ok=True)  # ensure folder
        output_track_video="output/tracks/tracked.mp4"
        tracked_video, csv_path = track_video(video_path, output_track_video)
        st.success("Tracking complete!")
        st.video(tracked_video)
        st.write(f"CSV saved at {csv_path}")
        
    # if st.button("Generate BEV Heatmap"):
    #     # Check weights file exists (bev_model expects monolayout_pretrained.ot)
    #     if not os.path.isfile("monolayout_pretrained.ot"):
    #         st.error("MonoLayout weights not found. Upload a .pt/.pth file above, or place it as 'monolayout_pretrained.ot' in the project folder.")
    #     else:
    #         try:
    #             bev_model = MonoLayoutBEV()
    #             cap = cv2.VideoCapture(video_path)
    #             ret, frame = cap.read()
    #             cap.release()

    #             if not ret:
    #                 st.error("could not read a frame from the uploaded video.")
    #             else:
    #                 bev = bev_model.infer_bev(frame)
    #                 plan_path = generate_floor_plan(bev)

    #                 mapped = map_tracks_to_bev(csv_path)
    #                 final_heatmap = generate_bev_heatmap(mapped, plan_path)

    #                 st.image(final_heatmap)
    #         except FileNotFoundError as e:
    #             st.error(str(e))
    #         except Exception as e:
    #             st.error(f"BEV generation failed: {e}")


    
st.markdown("---")
st.header("Heatmap on Floor Plan")

plan_file = st.file_uploader("Upload Floor Plan Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="plan_upload")
kernel_size = st.slider("Heatmap smoothing (kernel size)", min_value=5, max_value=101, value=35, step=2)
sigma = st.slider("Gaussian sigma (0 = auto)", min_value=0, max_value=50, value=0, step=1)
alpha = st.slider("Overlay opacity", min_value=0.1, max_value=0.95, value=0.6, step=0.05)

# Replace manual text coordinates with clickable picker
plan_path, dst_pts = pick_plan_points(plan_file, state_key="plan_points")  # NEW

if st.button("Generate Plan Heatmap") and plan_path and csv_path:
    if dst_pts is None or len(dst_pts) != 4:
        st.error("Select exactly 4 points on the plan before generating.")
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