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
        heat[y, x] += 1.0
    return heat

def _smooth_heatmap(heat: np.ndarray, kernel_size: int = 35, sigma: float = 0):
    k = max(1, int(kernel_size) | 1)  # ensure odd
    if k > 1:
        return cv2.GaussianBlur(heat, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return heat

def _colorize_and_overlay(plan_bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.6, colormap=cv2.COLORMAP_JET):
    # Normalize heat to 0..255
    if np.max(heat) > 0:
        heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)

    color = cv2.applyColorMap(heat_norm, colormap)

    # Masked overlay: only blend where heat > 0
    mask = heat_norm > 0
    overlay = plan_bgr.copy()
    overlay[mask] = cv2.addWeighted(plan_bgr[mask], 1 - alpha, color[mask], alpha, 0)
    return overlay, color, heat_norm

def generate_plan_heatmap_from_csv(
    csv_path: str,
    plan_image_path: str,
    src_points: np.ndarray,
    dst_points: np.ndarray | None = None,
    out_path: str | None = None,
    kernel_size: int = 35,
    sigma: float = 0,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
):
    """
    csv_path: tracking CSV (expects either columns cx,cy OR x,y,w,h OR x,y)
    plan_image_path: path to the floor plan image
    src_points: 4x2 points in the CCTV frame (from homography_config)
    dst_points: 4x2 points in the plan image (pixel coords). If None, uses plan corners.
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

    # Destination points default = image corners
    if dst_points is None:
        dst_points = np.array(
            [
                [0,           H_plan - 1],  # bottom-left
                [W_plan - 1,  H_plan - 1],  # bottom-right
                [W_plan - 1,  0],           # top-right
                [0,           0],           # top-left
            ],
            dtype=np.float32,
        )
    else:
        dst_points = np.asarray(dst_points, dtype=np.float32)
        if dst_points.shape != (4, 2):
            raise ValueError("dst_points must be 4x2 array")

    src_points = np.asarray(src_points, dtype=np.float32)
    if src_points.shape != (4, 2):
        raise ValueError("src_points must be 4x2 array")

    # Homography
    Hmat = cv2.getPerspectiveTransform(src_points, dst_points)

    # Load tracks
    df = pd.read_csv(csv_path)
    points_img = _select_points_from_df(df)  # Nx2

    # Warp points to plan pixel coords
    pts = points_img.reshape(-1, 1, 2).astype(np.float32)
    pts_plan = cv2.perspectiveTransform(pts, Hmat).reshape(-1, 2)

    # Accumulate heatmap
    heat = _accumulate_heatmap(pts_plan, H_plan, W_plan)
    heat = _smooth_heatmap(heat, kernel_size=kernel_size, sigma=sigma)

    # Overlay on plan
    overlay_bgr, color_bgr, heat_u8 = _colorize_and_overlay(plan_bgr, heat, alpha=alpha, colormap=colormap)

    # Save if requested
    if out_path:
        _ensure_dir(out_path)
        cv2.imwrite(out_path, overlay_bgr)

    # Return overlay (BGR), heat raw, and homography for debugging if needed
    return {
        "overlay_bgr": overlay_bgr,
        "heat": heat,
        "heat_u8": heat_u8,
        "color_bgr": color_bgr,
        "H": Hmat,
        "plan_size": (H_plan, W_plan),
        "points_plan": pts_plan,
    }

def parse_points_str(points_str: str):
    # Expose parser for Streamlit use
    return _parse_points_str(points_str)