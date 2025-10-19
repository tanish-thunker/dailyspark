import pandas as pd
from datetime import datetime

def append_dummy_video(csv_path="data/capsules.csv"):
    df = pd.read_csv(csv_path)
    max_id = int(df["id"].astype(int).max()) if not df.empty else 0
    new = {
        "id": max_id + 1,
        "interest": "programming",
        "subtopic": "python-basics",
        "format": "video",
        "title": "New Python Tips in 7 Minutes",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "duration_min": 7,
        "level": "beginner",
        "tags": "python;tips;basics",
        "language": "en",
        "source": "youtube",
        "quality_score": 0.78,
    }
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[{datetime.now().isoformat()}] appended 1 dummy item.")
