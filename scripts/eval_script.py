"""
Dice score and IoU on segmentation and ground truth masks
"""
import numpy as np
import torch
import nibabel as nib 
import os

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

def preproces_gt(gt_mask):
    gt_mask[gt_mask > 0] = 1
    gt_mask_resized = gt_mask[..., 8:-8, 8:-8] 
    return gt_mask_resized



testing_results = '../results/eval_on_test_samples'
testing_dir = '../data/small_test'

dice_vals = []
iou_vals = []

for t in os.listdir(testing_dir):
    gt_mask = nib.load(os.path.join(testing_dir, t, 'file name')).get_fdata()
    gt_mask = preproces_gt(gt_mask)
    seg_mask = torch.load(os.path.join(testing_results, t, 'file name'), map_location=torch.device('cpu'))
    seg_mask = seg_mask.numpy()
    seg_mask = np.squeeze(seg_mask)

    dice_val = dice_score(seg_mask, gt_mask)
    iou_val = iou(seg_mask, gt_mask)
    dice_vals.append(dice_val)
    iou_vals.append(iou_val)
    #visualize_masks(seg_mask, gt_mask)


avg_dice = np.mean(dice_vals)
avg_iou = np.mean(iou_vals)
print(f"Avg Dice score: {avg_dice:.4f}")
print(f"Avg IoU: {avg_iou:.4f}")