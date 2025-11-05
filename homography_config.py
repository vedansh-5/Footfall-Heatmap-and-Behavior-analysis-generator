import numpy as np

# 4 points on the FLOOR in the CCTV frame (pixel coordinates)
# --- THIS IS THE FIX ---
# The order is changed to a more intuitive Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left
src_points = np.array([
    [280, 400],   # top-left
    [1000, 400],  # top-right
    [960, 720],   # bottom-right
    [320, 720]    # bottom-left
], dtype=np.float32)

# Top-view BEV size (output heatmap resolution)
bev_width  = 1000
bev_height = 600

# Destination BEV plane
dst_points = np.array([
    [0,          bev_height],
    [bev_width,  bev_height],
    [bev_width,  0],
], dtype=np.float32)
