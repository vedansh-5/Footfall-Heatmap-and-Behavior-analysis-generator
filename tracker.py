import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import os

def resize_frame(frame, max_dim=640):
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))
    
def track_video(video_path, output_path):
    model = YOLO("yolov8n.pt")
    
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_tracks = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        detections = []
        
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = box.tolist()
                detections.append([[x1, y1, x2-x1, y2-y1], conf, int(cls)])

        tracks = tracker.update_tracks(detections, frame=frame)
        for t in tracks:
            if t.is_confirmed():
                bbox = t.to_ltwh()
                track_id = t.track_id
                cls = t.det_class
                x, y, w, h = bbox
                all_tracks.append([frame_num, track_id, cls, x, y, w, h])
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
                cv2.putText(frame, f"ID:{track_id}", (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        out.write(frame)
        frame_num += 5

    cap.release()
    out.release()

    # Save CSV
    df = pd.DataFrame(all_tracks, columns=["frame", "track_id", "class", "x", "y", "w", "h"])
    csv_path = os.path.join("output", "tracks.csv")
    df.to_csv(csv_path, index=False)
    print(f"Tracking complete, CSV saved at {csv_path}")
    return output_path, csv_path
