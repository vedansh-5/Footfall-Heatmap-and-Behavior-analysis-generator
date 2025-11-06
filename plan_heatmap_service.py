import os
import cv2
import numpy as np
from plan_heatmap import generate_plan_heatmap_from_csv

def run_and_save_plan_heatmap(csv_path: str, plan_path: str, src_points: np.ndarray, dst_pts: np.ndarray,
                              kernel_size: int, sigma: float, alpha: float):
    os.makedirs("output/plan_heatmap", exist_ok=True)
    out_overlay = "output/plan_heatmap/heatmap_overlay.png"
    result = generate_plan_heatmap_from_csv(
        csv_path=csv_path,
        plan_image_path=plan_path,
        src_points=src_points,
        dst_points=dst_pts,
        out_path=out_overlay,
        kernel_size=kernel_size,
        sigma=sigma,
        alpha=alpha,
    )
    cv2.imwrite("output/plan_heatmap/heat_u8.png", result["heat_u8"])
    cv2.imwrite("output/plan_heatmap/heat_color.png", result["color_bgr"])
    result["paths"] = {
        "overlay": out_overlay,
        "heat_u8": "output/plan_heatmap/heat_u8.png",
        "heat_color": "output/plan_heatmap/heat_color.png",
    }
    return result