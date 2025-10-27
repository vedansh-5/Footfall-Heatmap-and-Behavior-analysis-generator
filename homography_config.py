import numpy as np

# 4 points on the FLOOR in the CCTV frame (pixel coordinates)
# Pick floor corners from your video frame
src_points = np.array([
    [320, 720],  # bottom-left in CCTV view
    [960, 720],  # bottom-right
    [1000, 400], # top-right
    [280, 400]   # top-left
], dtype=np.float32)

# Top-view BEV size (output heatmap resolution)
bev_width  = 1000
bev_height = 600

# Destination BEV plane
dst_points = np.array([
    [0,          bev_height],
    [bev_width,  bev_height],
    [bev_width,  0],
    [0,          0]
], dtype=np.float32)
