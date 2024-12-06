import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pylab as plt

import nibabel as nib 
import os
import numpy as np


train_sample1_paths = [
    'full_training_eval_results/test_slices/BraTS20_Training_007_slice_23/BraTS20_Training_007_flair.nii',
    'full_training_eval_results/test_slices/BraTS20_Training_007_slice_23/BraTS20_Training_007_seg.nii',
    'full_training_eval_results/test_slices/BraTS20_Training_007_slice_23/BraTS20_Training_007_t1.nii',
    'full_training_eval_results/test_slices/BraTS20_Training_007_slice_23/BraTS20_Training_007_t1ce.nii',
    'full_training_eval_results/test_slices/BraTS20_Training_007_slice_23/BraTS20_Training_007_t2.nii',
]

plt.style.use('default')
fig, axes = plt.subplots(1, 6, figsize=(15,3))
train_samples = [train_sample1_paths]
for i, ax in enumerate(axes.reshape(-1)):
    j = i // 6
    k = i % 6
    if k == 5:
        seg_mask = torch.load(os.path.join('full_training_eval_results/ddim_smallset_750steps/BraTS20_Training_007_slice_23_output0.zip'), map_location=torch.device('cpu'))
        seg_mask = seg_mask.numpy()
        seg_mask = np.squeeze(seg_mask)
        seg_mask = seg_mask[0]
        ax.imshow(seg_mask)
    else:
        img = nib.load(train_samples[j][k]).get_fdata()
        ax.imshow(img)

plt.show()
