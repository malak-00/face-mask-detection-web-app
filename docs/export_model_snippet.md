# How to export the trained model to `.pth`

The original training notebook (`notebook59013e77e9.ipynb`) finishes
training but does NOT save the model. Add this cell at the end:

```python
import torch

# save just the weights (recommended - smaller file, version-safe)
torch.save(model.state_dict(), "mask_detector.pth")
print("model weights saved to mask_detector.pth")

# also save class order so we don't get confused later
import json
with open("classes.json", "w") as f:
    json.dump(train_dataset.classes, f)
print("class order:", train_dataset.classes)
```

Then download both files (`mask_detector.pth`, `classes.json`) from Colab
and drop `mask_detector.pth` into `app/` next to `app.py`.

## Reload sanity check

```python
from torchvision import models
from torch import nn
import torch

m = models.mobilenet_v2(weights=None)
m.classifier[1] = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2),
)
m.load_state_dict(torch.load("mask_detector.pth", map_location="cpu"))
m.train(False)
print("ok")
```
