import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pylab as plt

import nibabel as nib 

path = [
    '../data/testing/000246/brats_train_006_flair_080_w.nii',
    '../data/testing/000246/brats_train_006_t1_080_w.nii',
    '../data/testing/000246/brats_train_006_t1ce_080_w.nii',
    '../data/testing/000246/brats_train_006_t2_080_w.nii'
]
img = nib.load(path[3]).get_fdata()
print(img.shape)
print(f"The .nii files are stored in memory as numpy's: {type(img)}.")

plt.style.use('default')
fig, axes = plt.subplots(4, 1, figsize=(12,12))
for i, ax in enumerate(axes.reshape(-1)):
    img = nib.load(path[i]).get_fdata()
    ax.imshow(img)
plt.show()