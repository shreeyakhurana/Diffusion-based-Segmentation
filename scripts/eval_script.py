"""
Dice score and IoU on segmentation and ground truth masks
"""
import numpy as np
import torch
import nibabel as nib 
from scipy.ndimage import zoom

def visualize_masks(seg_mask, gt_mask):
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(seg_mask, cmap='gray')
    ax1.set_title('Segmentation Mask')
    ax1.axis('off')
    
    ax2.imshow(gt_mask, cmap='gray')
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')
    
    plt.tight_layout()

    plt.show()

def dice_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    return 2. * intersection.sum() / (mask1.sum() + mask2.sum())

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return intersection.sum() / union.sum()

def preproces_gt(gt_mask, target_size=(224, 224)):
    # convert any non zero value to 1
    gt_mask[gt_mask > 0] = 1
    # Calculate scaling factors for gt_mask
    scale_factors = (target_size[0] / gt_mask.shape[0],
                    target_size[1] / gt_mask.shape[1])
    
    # Resize gt_mask to match target size using scipy zoom
    gt_mask_resized = zoom(gt_mask, scale_factors, order=0)
    return gt_mask_resized



# Load the masks
seg_mask = torch.load(f'../results/eval_on_train_samples/000001_output0', map_location=torch.device('cpu'))
#convert to numpy
seg_mask = seg_mask.numpy()
seg_mask = np.squeeze(seg_mask)
gt_mask = nib.load('../data/training/000001/brats_train_001_seg_080_w.nii').get_fdata()
gt_mask = preproces_gt(gt_mask)

print(seg_mask.shape)
print(gt_mask.shape)


# Calculate dice score and IoU
dice_val = dice_score(seg_mask, gt_mask)
iou_val = iou(seg_mask, gt_mask)

print(f"Dice score: {dice_val:.4f}")
print(f"IoU: {iou_val:.4f}")
visualize_masks(seg_mask, gt_mask)