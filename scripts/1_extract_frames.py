import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
VIDEO_DIRS = {
    'real': 'data/real',
    'fake': 'data/fake'
}
OUTPUT_BASE = 'extracted_frames'
FRAMES_PER_VIDEO = 30  # Extract 30 frames per video
TARGET_SIZE = (224, 224)  # Resize to 224x224

def extract_frames_from_video(video_path, output_dir, num_frames=30):
    """Extract evenly spaced frames from a video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            num_frames = total_frames
        
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        extracted_count = 0
        for idx, frame_num in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Resize to reduce memory
            frame = cv2.resize(frame, TARGET_SIZE)
            
            # Save as JPEG (smaller than PNG)
            video_name = video_path.stem
            output_path = output_dir / f"{video_name}_frame_{idx:03d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            extracted_count += 1
        
        cap.release()
        return extracted_count
    
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0

def main():
    # Create output directories
    for label in ['real', 'fake']:
        os.makedirs(f"{OUTPUT_BASE}/{label}", exist_ok=True)
    
    total_frames = 0
    
    for label, video_dir in VIDEO_DIRS.items():
        print(f"\n{'='*50}")
        print(f"Processing {label.upper()} videos...")
        print(f"{'='*50}")
        
        video_path = Path(video_dir)
        output_path = Path(OUTPUT_BASE) / label
        
        if not video_path.exists():
            print(f"Directory not found: {video_dir}")
            continue
        
        video_files = list(video_path.glob('*.mp4')) + list(video_path.glob('*.avi'))
        
        if not video_files:
            print(f"No videos found in {video_dir}")
            continue
        
        for video_file in tqdm(video_files, desc=f"Extracting {label}"):
            frames_extracted = extract_frames_from_video(
                video_file, output_path, FRAMES_PER_VIDEO
            )
            total_frames += frames_extracted
    
    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"Total frames extracted: {total_frames}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
