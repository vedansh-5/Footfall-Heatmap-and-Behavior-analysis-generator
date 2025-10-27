import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import cm

def generate_bev_heatmap(mapped_df, floor_plan_path, save_path="output/bev/heatmap_on_plan.png"):
    plan = cv2.imread(floor_plan_path)
    heat = np.zeros((plan.shape[0], plan.shape[1]), dtype=np.float32)

    for _, row in mapped_df.iterrows():
        x, y = int(row["bev_x"]), int(row["bev_y"])
        if 0 <= x < heat.shape[1] and 0 <= y < heat.shape[0]:
            heat[y, x] += 1

    heat = gaussian_filter(heat, sigma=12)
    heat_norm = (heat / heat.max()) * 255
    heat_colored = cm.jet(heat_norm.astype(np.uint8))[:, :, :3]
    heat_colored = (heat_colored * 255).astype(np.uint8)

    overlay = cv2.addWeighted(plan, 0.6, heat_colored, 0.4, 0)
    cv2.imwrite(save_path, overlay)
    return save_path
