import json
import os

class MetadataAnnotator:
    """Generates a JSON report with all instance-level annotations."""
    
    def annotate(self, bboxes, class_names, output_path):
        """
        Args:
            bboxes: List of [x1, y1, x2, y2]
            class_names: List of strings (one per bbox)
            output_path: Path to save the JSON file
        """
        annotations = []
        for i, (bbox, cls) in enumerate(zip(bboxes, class_names)):
            annotations.append({
                "instance_id": i,
                "class": cls,
                "bbox": {
                    "x_min": round(bbox[0], 2),
                    "y_min": round(bbox[1], 2),
                    "x_max": round(bbox[2], 2),
                    "y_max": round(bbox[3], 2),
                }
            })
            
        report = {
            "num_instances": len(annotations),
            "annotations": annotations
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        return report
