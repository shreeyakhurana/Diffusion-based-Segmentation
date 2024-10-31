import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pylab as plt

import nibabel as nib 
import os
import numpy as np

tensor = torch.load("eval_on_train_samples/000001_output0", map_location=torch.device('cpu'))
print(type(tensor), tensor.shape)
print(tensor[0][0][0])

test_sample1_paths = [
    '../data/testing/000246/brats_train_006_flair_080_w.nii',
    '../data/testing/000246/brats_train_006_t1_080_w.nii',
    '../data/testing/000246/brats_train_006_t1ce_080_w.nii',
    '../data/testing/000246/brats_train_006_t2_080_w.nii'
]
train_sample1_paths = [
    '../midpoint_eval_results/midpoint_eval_samples/BraTS20_Training_010_slice_2/BraTS20_Training_010_flair.nii.gz',
    '../midpoint_eval_results/midpoint_eval_samples/BraTS20_Training_010_slice_2/BraTS20_Training_010_seg.nii.gz',
    '../midpoint_eval_results/midpoint_eval_samples/BraTS20_Training_010_slice_2/BraTS20_Training_010_t1.nii.gz',
    '../midpoint_eval_results/midpoint_eval_samples/BraTS20_Training_010_slice_2/BraTS20_Training_010_t1ce.nii.gz',
    '../midpoint_eval_results/midpoint_eval_samples/BraTS20_Training_010_slice_2/BraTS20_Training_010_t2.nii.gz',
]
train_sample2_paths = [
    '../data/training/000002/brats_train_001_flair_081_w.nii.gz',
    '../data/training/000002/brats_train_001_t1_081_w.nii.gz',
    '../data/training/000002/brats_train_001_t1ce_081_w.nii.gz',
    '../data/training/000002/brats_train_001_t2_081_w.nii.gz',
    '../data/training/000002/brats_train_001_seg_081_w.nii.gz',
]
img = nib.load(train_sample1_paths[3]).get_fdata()
print(img.shape)
print(f"The .nii files are stored in memory as numpy's: {type(img)}.")

plt.style.use('default')
fig, axes = plt.subplots(1, 6, figsize=(15,3))
train_samples = [train_sample2_paths]
for i, ax in enumerate(axes.reshape(-1)):
    j = i // 6
    k = i % 6
    if k == 5:
        seg_mask = torch.load(os.path.join('eval_on_train_samples/000001_output0'), map_location=torch.device('cpu'))
        seg_mask = seg_mask.numpy()
        seg_mask = np.squeeze(seg_mask)
        ax.imshow(seg_mask)
    else:
        img = nib.load(train_samples[j][k]).get_fdata()
        ax.imshow(img)

# seg_mask = torch.load(os.path.join('results/eval_on_train_samples/000001_output0'), map_location=torch.device('cpu'))
# seg_mask = seg_mask.numpy()
# seg_mask = np.squeeze(seg_mask)
# ax.imshow(seg_mask)
plt.show()