import numpy as np

src_points = np.array([
    [280, 400],
    [1000, 400],
    [960, 720],
    [320, 720]
], dtype=np.float32)

bev_width  = 1000
bev_height = 600

# Destination BEV plane
dst_points = np.array([
    [0,          bev_height],
    [bev_width,  bev_height],
    [bev_width,  0],
], dtype=np.float32)
