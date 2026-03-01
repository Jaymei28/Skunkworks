from ultralytics import YOLO
import os

def train():
    dataset_yaml = "c:/Users/Jaymei/.gemini/antigravity/scratch/pytorch3d_renderer/dataset_apple/data.yaml"
    model_name = "yolov8n.pt"
    
    print(f"Starting YOLOv8 training on {dataset_yaml}...")
    
    # Load a pretrained model
    model = YOLO(model_name)
    
    # Train the model
    # workers=2, batch=8 for Windows stability
    results = model.train(
        data=dataset_yaml,
        epochs=5,
        imgsz=640,
        batch=8,
        workers=2,
        device=0
    )
    
    print("\nTraining complete!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == "__main__":
    train()
