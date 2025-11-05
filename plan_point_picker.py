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
    This version uses the 'streamlit-image-coordinates' library and correctly scales the points.
    """
    if plan_file is None:
        return None, None

    if state_key not in st.session_state:
        st.session_state[state_key] = []

    plan_path = _save_uploaded_plan(plan_file)
    img = Image.open(plan_path).convert("RGB")
    W, H = img.size

    st.info("Click 4 points on the plan image below. The coordinates will be recorded in order.")

    # --- THIS IS THE FIX ---
    # Define the display width and calculate the scaling factor.
    display_width = 700
    scale_factor = W / display_width
    # Calculate the corresponding display height.
    display_height = H / scale_factor

    # Display the image and get coordinates on click.
    value = streamlit_image_coordinates(img, width=display_width, key=f"{state_key}_picker")

    if value:
        # Correctly scale the coordinates from the displayed image back to the original image size.
        # The scaling factor is the same for both axes because aspect ratio is preserved.
        original_x = value['x'] * scale_factor
        original_y = value['y'] * scale_factor
        
        scaled_point = {'x': original_x, 'y': original_y}

        # Check if the new click is different from the last one to avoid duplicates on rerun.
        if not st.session_state[state_key] or scaled_point != st.session_state[state_key][-1]:
            if len(st.session_state[state_key]) < 4:
                st.session_state[state_key].append(scaled_point)
                st.rerun()

    picked_points = st.session_state.get(state_key, [])
    if picked_points:
        st.write("Selected Points (Original Image Coordinates):")
        for i, point in enumerate(picked_points, 1):
            st.write(f"Point {i}: ({int(point['x'])}, {int(point['y'])})")

    if st.button("Clear Points"):
        st.session_state[state_key] = []
        st.rerun()

    if len(picked_points) == 4:
        st.success("4 points selected.")
        points_array = np.array([[p['x'], p['y']] for p in picked_points], dtype=np.float32)
        return plan_path, points_array
    else:
        st.warning(f"Please select {4 - len(picked_points)} more point(s).")
        return plan_path, None

