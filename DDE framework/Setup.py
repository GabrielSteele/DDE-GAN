# conda install python==3.10.14

# pip install simpleitk==2.1.1.1 opencv-python tensorboard==2.10.1 scipy==1.12.0  scikit-image==0.20.0 pandas==1.5.3 matplotlib==3.8.0 nibabel==5.2.0 kornia numpy==1.26.4
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install https://github.com/odlgroup/odl/archive/master.zip
# conda install -c astra-toolbox astra-toolbox

# conda env create -nProject -f \Users\2418015\OneDrive - University of Dundee\Documents\4th Year\Project\Code\pythonProject\Dual-domain Framework\Project.yml
# conda env export > name.yml

# ValueError: Since image dtype is floating point, you must specify the data_range parameter. Please read the documentation carefully (including the note). It is recommended that you always specify the data_range anyway.
# Exception ignored in: <function AstraCudaImpl.__del__ at 0x000001AD1D3B39A0>
## Solution >> utils.common get_ssim, line80: out+=structural_similarity(img1[i].squeeze(),img2[i].squeeze(), data_range=255)

#MIGHT WANT TO ADJUST SSIM

#RuntimeError: Input type (double) and bias type (float) should be the same
## Solution >> dataset.dataset_lits_train Line22: pet, ct = np.float32(PET[slice]), np.float32(CT[slice])

## OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
## Solution: envs>Project>Library>bin delete libiomp5md.dll

# AttributeError: module 'odl.contrib.torch' has no attribute 'OperatorAsModule'. Did you mean: 'OperatorModule'?
# Just change 'OperatorAsModule' to 'OperatorModule' (lines 35 and 19 in .Projection_operator)


# RuntimeError: Input type (double) and bias type (float) should be the same
# Solution: In Model.model the conv block class, in the forward def change out = ... x > x.float()

# Error: append is outdated
# Solution: utils>logger.py  change self.log = self.log.append(tmp, ignore_index=True) to this: self.log = pd.concat([self.log, tmp], ignore_index=True)

# RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 176 but got size 177 for tensor number 1 in the list.
# Solution:

import torch
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
import os
import torch.nn as nn
from skimage.metrics import normalized_root_mse
from collections import OrderedDict
import warnings
from skimage.metrics import structural_similarity
import cv2
from torch.utils.data import DataLoader




