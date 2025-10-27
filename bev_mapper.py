import pandas as pd
import numpy as np

def map_tracks_to_bev(csv_path, bev_width=800, bev_height=800):
    df = pd.read_csv(csv_path)
    df["cx"] = df["x"] + df["w"] / 2
    df["cy"] = df["y"] + df["h"]
    df["bev_x"] = (df["cx"] / df["cx"].max()) * bev_width
    df["bev_y"] = (df["cy"] / df["cy"].max()) * bev_height
    return df[["frame", "track_id", "bev_x", "bev_y"]]
