import os
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import streamlit as st

# Use a different, simpler library for getting coordinates
from streamlit_image_coordinates import streamlit_image_coordinates

def _save_uploaded_plan(plan_file) -> str:
    """Saves the uploaded file to the 'input_plans' directory."""
    os.makedirs("input_plans", exist_ok=True)
    plan_path = os.path.join("input_plans", plan_file.name)
    with open(plan_path, "wb") as f:
        f.write(plan_file.getbuffer())
    return plan_path

def pick_plan_points(plan_file, state_key: str = "plan_pts") -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Displays the plan and allows the user to pick 4 points by clicking.
    This version uses the 'streamlit-image-coordinates' library.
    """
    if plan_file is None:
        return None, None

    # Initialize the list of points in the session state if it doesn't exist
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    plan_path = _save_uploaded_plan(plan_file)
    img = Image.open(plan_path).convert("RGB")
    W, H = img.size

    st.info("Click 4 points on the plan image below. The coordinates will be recorded in order.")

    # Display the image and get coordinates on click
    value = streamlit_image_coordinates(img, key=f"{state_key}_picker")

    # If the user clicked, 'value' will contain the coordinates
    if value:
        # Check if the new click is different from the last one to avoid duplicates on rerun
        if not st.session_state[state_key] or value != st.session_state[state_key][-1]:
            if len(st.session_state[state_key]) < 4:
                st.session_state[state_key].append(value)
                # We need to rerun to update the display of points immediately
                st.rerun()

    # Display the points that have been collected so far
    picked_points = st.session_state.get(state_key, [])
    if picked_points:
        st.write("Selected Points (in order of clicking):")
        for i, point in enumerate(picked_points, 1):
            st.write(f"Point {i}: ({point['x']}, {point['y']})")

    # Add a button to clear the collected points
    if st.button("Clear Points"):
        st.session_state[state_key] = []
        st.rerun()

    # Once 4 points are collected, return them
    if len(picked_points) == 4:
        st.success("4 points selected.")
        # Convert the list of dicts to the required numpy array format
        points_array = np.array([[p['x'], p['y']] for p in picked_points], dtype=np.float32)
        return plan_path, points_array
    else:
        st.warning(f"Please select {4 - len(picked_points)} more point(s).")
        return plan_path, None

