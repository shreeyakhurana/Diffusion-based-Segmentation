import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pylab as plt

import nibabel as nib 

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
    '../data/training/000001/brats_train_001_flair_080_w.nii.gz',
    '../data/training/000001/brats_train_001_t1_080_w.nii.gz',
    '../data/training/000001/brats_train_001_t1ce_080_w.nii.gz',
    '../data/training/000001/brats_train_001_t2_080_w.nii.gz',
    '../data/training/000001/brats_train_001_seg_080_w.nii.gz',
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
fig, axes = plt.subplots(2, 5, figsize=(12,12))
train_samples = [train_sample1_paths, train_sample2_paths]
for i, ax in enumerate(axes.reshape(-1)):
    j = i // 5
    k = i % 5
    img = nib.load(train_samples[j][k]).get_fdata()
    ax.imshow(img)
plt.show()