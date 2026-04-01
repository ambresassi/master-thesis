import os
from pathlib import Path
import gzip 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random

import torch

# 1. Load RGB, seg, depth
# 2. For seg: extract R channel if needed
# 3. For seg: remap labels to 5 classes + ignore
# 4. Resize RGB / seg / depth together
# 5. Crop RGB / seg / depth together
# 6. Convert all to tensors : final output for one id : RGB: [0,1] (normalized with ImageNet stats good start, can change later (find source)), seg: long tensor, depth: [0,1] or [0,255] or float ?  depending on how it's normalized or not.
# 7. Normalize RGB only
# 8. Optionally build valid mask for depth -> see synthtic VS real unit mismatch in depth 