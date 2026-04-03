import cv2
import argparse
import sys
from pathlib import Path

def extract_frames(video_path: str, fps_target: int = 1):
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: {video_path} does not exist.")
        sys.exit(1)
        
    output_dir = Path("data/raw_frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing '{video_file.name}' | Base FPS: {fps:.2f} | Total Frames: {total_frames}")
    
    # คำนวณช่วงระยะห่างในการสกัดเฟรม (Interval)
    frame_interval = int(fps / fps_target) if fps_target < fps else 1
    if frame_interval < 1:
        frame_interval = 1
        
    saved_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"{video_file.stem}_{frame_idx}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames...")
                
        frame_idx += 1
        
    cap.release()
    print(f"✅ Extraction complete! Saved {saved_count} frames to {output_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 video for Model Training Dataset Generation")
    parser.add_argument("video_path", type=str, help="Path to input video file (e.g. data/video.mp4)")
    parser.add_argument("--fps", type=int, default=1, help="Target FPS to extract (default: 1 frame per second)")
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.fps)
