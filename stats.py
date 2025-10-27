import pandas as pd

def get_stats(csv_path):
    df = pd.read_csv(csv_path)
    unique_ids = df["track_id"].nunique()

    class_counts = df["class"].value_counts().to_dict()
    total_people = class_counts.get(0,0)
    
    return {
        "total_unique_objects": unique_ids,
        "class_counts": class_counts,
        "total_people": total_people
        
    }