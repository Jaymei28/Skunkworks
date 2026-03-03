import sys
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    import pytorch3d
    print(f"PyTorch3D version: {pytorch3d.__version__}")
    from pytorch3d.structures import Meshes
    print("Successfully imported PyTorch3D Meshes!")
except Exception as e:
    import traceback
    traceback.print_exc()
