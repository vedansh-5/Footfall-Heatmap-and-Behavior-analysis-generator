import os
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import streamlit as st

from streamlit_image_coordinates import streamlit_image_coordinates

def _save_uploaded_plan(plan_file) -> str:
    os.makedirs("input_plans", exist_ok=True)
    plan_path = os.path.join("input_plans", plan_file.name)
    with open(plan_path, "wb") as f:
        f.write(plan_file.getbuffer())
    return plan_path

def pick_plan_points(plan_file, state_key: str = "plan_pts") -> Tuple[Optional[str], Optional[np.ndarray]]:
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    plan_path = None
    if plan_file:
        plan_path = _save_uploaded_plan(plan_file)
        pil_image = Image.open(plan_path)
        
        value = streamlit_image_coordinates(pil_image, key=f"picker_{state_key}")

        if value:
            point = value
            if point not in st.session_state[state_key]:
                if len(st.session_state[state_key]) < 4:
                    st.session_state[state_key].append(point)
                else:
                    st.warning("Maximum of 4 points reached. Clear points to start over.")
                st.rerun()

    picked_points = st.session_state[state_key]
    if picked_points:
        st.write("Picked points:")
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

