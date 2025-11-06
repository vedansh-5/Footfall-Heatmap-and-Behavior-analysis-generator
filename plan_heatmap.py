import os
import cv2
import numpy as np
import pandas as pd

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _parse_points_str(points_str: str):
    # "x1,y1; x2,y2; x3,y3; x4,y4" -> np.array shape (4,2)
    pts = []
    for token in points_str.replace("\n", " ").split(";"):
        token = token.strip()
        if not token:
            continue
        x_str, y_str = token.split(",")
        pts.append([float(x_str), float(y_str)])
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Need exactly 4 points in 'x,y; x,y; x,y; x,y' format")
    return pts

def _select_points_from_df(df: pd.DataFrame):
    # Return Nx2 array of foot positions (image coords) inferred from CSV columns
    cols = set(c.lower() for c in df.columns)

    # Normalize column names
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"frame", "frames"}: rename_map[c] = "frame"
        if lc in {"id", "track_id", "trackid"}: rename_map[c] = "track_id"
        if lc in {"x", "left"}: rename_map[c] = "x"
        if lc in {"y", "top"}: rename_map[c] = "y"
        if lc in {"w", "width"}: rename_map[c] = "w"
        if lc in {"h", "height"}: rename_map[c] = "h"
        if lc in {"cx", "center_x", "centrex"}: rename_map[c] = "cx"
        if lc in {"cy", "center_y", "centrey"}: rename_map[c] = "cy"
    if rename_map:
        df = df.rename(columns=rename_map)
        cols = set(df.columns)

    points = None
    if {"cx", "cy"}.issubset(cols):
        # Use provided centers (assumed foot position if already at feet)
        points = df[["cx", "cy"]].to_numpy(np.float32)
    elif {"x", "y", "w", "h"}.issubset(cols):
        # Use bbox bottom-center as foot position
        cx = df["x"].to_numpy(np.float32) + df["w"].to_numpy(np.float32) / 2.0
        cy = df["y"].to_numpy(np.float32) + df["h"].to_numpy(np.float32)
        points = np.stack([cx, cy], axis=1)
    elif {"x", "y"}.issubset(cols):
        # If only x,y are present, assume they already represent the foot point
        points = df[["x", "y"]].to_numpy(np.float32)
    else:
        raise ValueError(
            "CSV must contain either (cx, cy) or (x, y, w, h) or (x, y). "
            f"Found columns: {list(df.columns)}"
        )
    return points

def _accumulate_heatmap(points_plan: np.ndarray, H: int, W: int):
    heat = np.zeros((H, W), dtype=np.float32)
    # Round and accumulate
    xs = np.clip(np.round(points_plan[:, 0]).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(points_plan[:, 1]).astype(np.int32), 0, H - 1)
    for x, y in zip(xs, ys):
        # Decrease the accumulation value to prevent oversaturation in one spot
        heat[y, x] += 25.0
    return heat

def _smooth_heatmap(heat: np.ndarray, kernel_size: int = 35, sigma: float = 0):
    # Ensure kernel is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(heat, (kernel_size, kernel_size), sigma)


def _colorize_and_overlay(plan_bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.6, colormap=cv2.COLORMAP_JET):
    # Normalize the smoothed float heatmap to the 0-255 range
    heat_u8 = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply colormap to the normalized 8-bit heatmap
    color_bgr = cv2.applyColorMap(heat_u8, colormap)
    
    # Make areas with no heat transparent in the color map
    color_bgr[heat_u8 == 0] = 0
    
    # Blend the colorized heatmap with the original plan
    overlay_bgr = cv2.addWeighted(plan_bgr, 1 - alpha, color_bgr, alpha, 0)
    
    return {
        "overlay_bgr": overlay_bgr,
        "heat_u8": heat_u8,
        "color_bgr": color_bgr,
    }


def generate_plan_heatmap_from_csv(
    csv_path: str,
    plan_image_path: str,
    src_points: np.ndarray,
    dst_points: np.ndarray,  # The default value is removed. This is now a required argument.
    out_path: str | None = None,
    kernel_size: int = 35,
    sigma: float = 0,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
):
    """
    Generates a heatmap on a floor plan using a homography transformation.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isfile(plan_image_path):
        raise FileNotFoundError(f"Plan image not found: {plan_image_path}")

    # Load plan image (BGR)
    plan_bgr = cv2.imread(plan_image_path, cv2.IMREAD_COLOR)
    if plan_bgr is None:
        raise RuntimeError(f"Failed to read plan image: {plan_image_path}")
    H_plan, W_plan = plan_bgr.shape[:2]

    # --- THIS IS THE FIX ---
    # All previous flipping logic has been removed.
    # The transformation now directly maps the source points from the video
    # to the destination points you click on the plan.
    Hmat = cv2.getPerspectiveTransform(src_points, dst_points)
    # --- END FIX ---

    # 1. Select points from CSV and transform them
    df = pd.read_csv(csv_path)
    points_img = _select_points_from_df(df)  # Nx2

    # Warp points to plan pixel coords
    pts = points_img.reshape(-1, 1, 2).astype(np.float32)
    pts_plan = cv2.perspectiveTransform(pts, Hmat).reshape(-1, 2)

    # --- THIS IS THE FIX ---
    # The bounding box for filtering must be calculated from the original destination points,
    # as there is no longer a flipped coordinate system.
    x_min, y_min = np.min(dst_points, axis=0)
    x_max, y_max = np.max(dst_points, axis=0)
    
    inside_mask = (
        (pts_plan[:, 0] >= x_min) &
        (pts_plan[:, 0] <= x_max) &
        (pts_plan[:, 1] >= y_min) &
        (pts_plan[:, 1] <= y_max)
    )
    pts_plan_filtered = pts_plan[inside_mask]
    # --- END FIX ---

    # 2. Accumulate points onto a blank canvas (use the filtered points)
    heat_raw = _accumulate_heatmap(pts_plan_filtered, H_plan, W_plan)

    # 3. Smooth the raw heatmap
    heat_smooth = _smooth_heatmap(heat_raw, kernel_size, sigma)

    # 4. Colorize and overlay onto the plan
    # --- THIS IS THE FIX for the color ---
    # Change the colormap back to JET, which provides the Red -> Yellow -> Blue scale.
    result = _colorize_and_overlay(plan_bgr, heat_smooth, alpha, colormap=cv2.COLORMAP_JET)
    # --- END FIX ---

    # 5. Save and return
    if out_path:
        _ensure_dir(out_path)
        cv2.imwrite(out_path, result["overlay_bgr"])

    # Return overlay (BGR) and other artifacts for debugging
    return {
        "overlay_bgr": result["overlay_bgr"],
        "heat": heat_raw,
        "heat_u8": result["heat_u8"],
        "color_bgr": result["color_bgr"],
        "H": Hmat,
        "plan_size": (H_plan, W_plan),
        "points_plan": pts_plan,
    }

def parse_points_str(points_str: str):
    # Expose parser for Streamlit use
    return _parse_points_str(points_str)