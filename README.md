# PyTorch3D Rendering Library

A modular library for generating synthetic 3D data using PyTorch3D.

## Features
- **Easy Import**: Support for OBJ and PLY files via `MeshLoader`.
- **Customizable Randomizers**: Modular system for randomizing poses, lighting, textures, etc.
- **Multiple Annotation Types**: Out-of-the-box support for:
    - RGB Renders
    - Depth Maps
    - Semantic Masks
    - 2D Bounding Boxes

## Installation

Ensure you have PyTorch and PyTorch3D installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from renderer.core import Renderer3D
from renderer.loader import MeshLoader
from renderer.randomizers.pose import PoseRandomizer
from renderer.annotators.common import DepthAnnotator

# Initialize
renderer = Renderer3D(image_size=512)
mesh = MeshLoader.load("model.obj")

# Randomize
PoseRandomizer().apply(renderer)

# Render & Annotate
image = renderer.render(mesh)
depth = DepthAnnotator().annotate(renderer, mesh)
```

## Directory Structure
- `renderer/`: Core library code.
- `renderer/randomizers/`: Add your own randomization logic here.
- `renderer/annotators/`: Add new annotation types (e.g. keypoints, normals).
- `examples/`: Guided examples for data generation pipelines.
