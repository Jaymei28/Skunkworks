from ultralytics import YOLO
import cv2
import os
import sys

def main():
    # Load the trained model
    model_path = r"c:\Users\Jaymei\.gemini\antigravity\scratch\pytorch3d_renderer\dataset_apple\runs\detect\train2\weights\best.pt"
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    model = YOLO(model_path)
    
    # Path to input video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = r"c:\Users\Jaymei\Desktop\Objects\apple\WIN_20260211_10_13_36_Pro.mp4"
    
    video_path = video_path.strip('"')
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video path
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(video_dir, f"{video_name}_detected.mp4")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("\n" + "="*60)
    print("🎬 VIDEO DETECTION STARTED")
    print("="*60)
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print("="*60 + "\n")
    
    frame_count = 0
    
    print("Press 'q' during playback to stop processing early")
    print("Processing video...\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection - Updated to use GPU (device=0) and larger imgsz
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            device=0, # Use GPU
            verbose=False
        )
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Write to output video
        out.write(annotated_frame)
        
        # Display live preview
        cv2.imshow('Video Processing - Press Q to stop', annotated_frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Processing stopped by user")
            break
        
        # Show progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ VIDEO DETECTION COMPLETE")
    print(f"📁 Output saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
