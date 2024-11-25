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
    # if union is 0, return 1
    if mask1.sum() + mask2.sum() == 0:
        return 1.0
    return 2. * intersection.sum() / (mask1.sum() + mask2.sum())

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if union.sum() == 0:
        return 1.0
    return intersection.sum() / union.sum()

def preproces_gt(gt_mask):
    gt_mask[gt_mask > 0] = 1
    #gt_mask_resized = gt_mask[..., 8:-8, 8:-8] 
    gt_mask_resized = gt_mask
    return gt_mask_resized

def preprocess_seg(seg_mask):
    seg_mask[seg_mask > 0.8] = 1
    seg_mask[seg_mask < 0.4] = 0
    #seg_mask_resized = seg_mask[..., 8:-8, 8:-8]
    seg_mask_resized = seg_mask
    return seg_mask_resized



seg_pred_dir = '../full_training_eval_results/ddim_results/'
ground_truth_dir = '../full_training_eval_results/test_slices/'

dice_vals = []
iou_vals = []

for t in os.listdir(ground_truth_dir):
    base_name = t.split('_slice')[0] #BraTS20_Training_007
    print(f"Processing {t}")
    gt_file_name = os.path.join(ground_truth_dir, t, base_name+"_seg.nii")
    print(f"Ground truth file: {gt_file_name}")
    gt_mask = nib.load(gt_file_name).get_fdata()
    gt_mask = preproces_gt(gt_mask)
    
    seg_file_name = os.path.join(seg_pred_dir, t+"_output0")
    print(f"Segmentation file: {seg_file_name}")
    seg_mask = torch.load(seg_file_name, map_location=torch.device('cpu'))
    seg_mask = seg_mask.numpy()
    seg_mask = np.squeeze(seg_mask)
    seg_mask = preprocess_seg(seg_mask)
    seg_mask = seg_mask[0]


    dice_val = dice_score(seg_mask, gt_mask)
    iou_val = iou(seg_mask, gt_mask)
    dice_vals.append(dice_val)
    iou_vals.append(iou_val)
    # visualize_masks(seg_mask, gt_mask)
    # breakpoint()


avg_dice = np.mean(dice_vals)
avg_iou = np.mean(iou_vals)
print(f"Avg Dice score: {avg_dice:.4f}")
print(f"Avg IoU: {avg_iou:.4f}")