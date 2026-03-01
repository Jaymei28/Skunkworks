import json
import os
import glob
from tqdm import tqdm

def convert_to_yolo():
    # Paths
    dataset_dir = "c:/Users/Jaymei/.gemini/antigravity/scratch/pytorch3d_renderer/dataset_apple"
    json_dir = os.path.join(dataset_dir, "labels")
    yolo_dir = os.path.join(dataset_dir, "yolo_labels")
    os.makedirs(yolo_dir, exist_ok=True)
    
    img_width = 512
    img_height = 512
    
    # Class mapping
    classes = {"Apple": 0}
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files. Converting to YOLO format...")
    
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        yolo_lines = []
        for ann in data["annotations"]:
            cls_name = ann["class"]
            cls_id = classes.get(cls_name, 0) # Default to 0 if not found
            
            bbox = ann["bbox"]
            x_min = bbox["x_min"]
            y_min = bbox["y_min"]
            x_max = bbox["x_max"]
            y_max = bbox["y_max"]
            
            # Calculate YOLO coordinates    
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Clip and format
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        # Save to .txt
        filename = os.path.basename(json_file).replace(".json", ".txt").replace("meta_", "apple_")
        with open(os.path.join(yolo_dir, filename), 'w') as f:
            f.write("\n".join(yolo_lines))
            
    # Create dataset.yaml
    yaml_content = f"""
path: {dataset_dir}
train: images
val: images

names:
  0: Apple
"""
    with open(os.path.join(dataset_dir, "data.yaml"), 'w') as f:
        f.write(yaml_content.strip())
        
    import shutil
    # Move original JSONs out of the way
    json_backup = os.path.join(dataset_dir, "labels_json")
    if os.path.exists(json_backup):
        shutil.rmtree(json_backup)
    os.rename(json_dir, json_backup)
    
    # Move YOLO labels to the 'labels' path
    os.rename(yolo_dir, json_dir)
        
    print(f"\nConversion complete! YOLO labels (txt) are now in: {json_dir}")
    print(f"Original JSON labels backed up to: {json_backup}")
    print(f"Dataset YAML created at: {os.path.join(dataset_dir, 'data.yaml')}")

if __name__ == "__main__":
    convert_to_yolo()
