# multimodal-medical-diagnosis

Multi-modal deep learning baseline for pneumonia diagnosis that fuses chest X-ray images with clinical tabular data using PyTorch.

## Features
- CNN image encoder for chest X-ray inputs
- MLP encoder for clinical data
- Fusion classifier for pneumonia prediction
- Grad-CAM support for explainability on the image branch

## Quick start
```python
import torch
from multimodal_medical_diagnosis import MultiModalPneumoniaModel, GradCAM

model = MultiModalPneumoniaModel(clinical_input_dim=8, num_classes=2)
images = torch.randn(4, 1, 128, 128)
clinical = torch.randn(4, 8)

logits = model(images, clinical)

gradcam = GradCAM(model, model.image_encoder.features[6])
heatmaps = gradcam.generate(images, clinical)
gradcam.close()
```

## Run tests
```bash
python -m unittest discover -s tests -v
```
