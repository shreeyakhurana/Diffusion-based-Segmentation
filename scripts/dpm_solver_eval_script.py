"""
Evaluation Utils for DPM-Solver++ Experiments
Dice score and IoU on segmentation and ground truth masks
"""
import numpy as np
import torch
import nibabel as nib 
import os
import matplotlib.pyplot as plt

def visualize_masks(seg_mask, gt_mask, samples, include_input=False):
    
    
    if not include_input:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(seg_mask, cmap='gray')
        ax1.set_title('Segmentation Mask')
        ax1.axis('off')
        
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title('Ground Truth Mask')
        ax2.axis('off')
        
        plt.tight_layout()

        plt.show()
    else:
        fig, axes = plt.subplots(1, 6, figsize=(10, 5))
        
        inputs = []

        for suffix in ["_t1.nii", "_t1ce.nii", "_t2.nii", "_flair.nii"]:
            input1 = os.path.join(ground_truth_dir, t, base_name+suffix)
            input1 = nib.load(input1).get_fdata()
            inputs.append(input1)
        
        for i in range(4):
            axes[i].imshow(inputs[i], cmap="gray")
            axes[i].axis('off')

        axes[4].imshow(seg_mask, cmap='gray')
        axes[4].set_title('Predicted Mask')
        axes[4].axis('off')
        
        axes[5].imshow(gt_mask, cmap='gray')
        axes[5].set_title('Ground Truth Mask')
        axes[5].axis('off')
        
        plt.tight_layout()

        plt.show()

def visualize_figure(dirs, processed_gt_mask):
    masks = []
    for d in dirs:
        i = 0
        for t in os.listdir(ground_truth_dir):
            if t.startswith("."): continue
            if i == 1:
                seg_file_name = os.path.join(d, t+"_output.zip")
                seg_mask = torch.load(seg_file_name, map_location=torch.device('cpu'))
                seg_mask = seg_mask.numpy()
                seg_mask = np.squeeze(seg_mask)
                processed_seg_mask, _, _ = preprocess_seg(seg_mask)
                masks.append(processed_seg_mask)
            i += 1

    fig, axes = plt.subplots(1, 5, figsize=(10, 5))

    axes[0].imshow(masks[0], cmap='gray')
    axes[0].set_title('Ensemble=1')
    axes[0].axis('off')

    axes[1].imshow(masks[1], cmap='gray')
    axes[1].set_title('Ensemble=5')
    axes[1].axis('off')

    axes[2].imshow(masks[2], cmap='gray')
    axes[2].set_title('Ensemble=10')
    axes[2].axis('off')

    axes[3].imshow(masks[3], cmap='gray')
    axes[3].set_title('Ensemble=25')
    axes[3].axis('off')  

    axes[4].imshow(processed_gt_mask, cmap='gray')
    axes[4].set_title('Ground Truth Mask')
    axes[4].axis('off')

    
    plt.tight_layout()

    plt.show()

def dice_score_old(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    # if union is 0, return 1
    if mask1.sum() + mask2.sum() == 0:
        return 1.0
    return 2. * intersection.sum() / (mask1.sum() + mask2.sum())

def dice_score(pred, targs):
    pred = torch.tensor(pred)
    targs = torch.tensor(targs)
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if union.sum() == 0:
        return 1.0
    return intersection.sum() / union.sum()

def preproces_gt(gt_mask):
    gt_mask[gt_mask > 0] = 1
    return gt_mask

def preprocess_seg(seg_mask):
    # try normalizing to 0 - 1
    seg_mask = (seg_mask-np.min(seg_mask))/(np.max(seg_mask)-np.min(seg_mask))

    avg_pixel = np.mean(seg_mask)
    std_pixel = np.std(seg_mask)

    seg_mask[seg_mask > 0.8] = 1 
    # The stuff in between we want to move to white (1) if the avg pixel is > THRESH else to 0
    # if avg_pixel < 0.12:
    #     seg_mask[(0.4 < seg_mask) & (seg_mask <= 0.8)] = 1
    # else:
    #     seg_mask[(0.4 < seg_mask) & (seg_mask <= 0.8)] = 0
    seg_mask[seg_mask < 0.4] = 0
    return seg_mask, avg_pixel, std_pixel



# seg_pred_dir = '../results/dpm_solver++_50steps_ensemble10/'
# ground_truth_dir = '../data/test_slices_20/'

seg_pred_dir = '../results/ddpm_results_1000steps_ensemble1/'
ground_truth_dir = '../data/test_slices_20/'

# seg_pred_dir = '../results/dpm_solver++_50steps/'
# ground_truth_dir = '../data/test_slices_200/'

dice_vals = []
iou_vals = []

# Statistics about the predicted masks after normalizing before thresholding
avgs = []
stds = []
black_or_not = []

idx = 0
for t in os.listdir(ground_truth_dir):
    if t.startswith("."): continue
    print(t)
    base_name = t.split('_slice')[0] #BraTS20_Training_007
    print(f"Processing {t}")
    gt_file_name = os.path.join(ground_truth_dir, t, base_name+"_seg.nii")
    print(f"Ground truth file: {gt_file_name}")
    gt_mask = nib.load(gt_file_name).get_fdata()
    processed_gt_mask = preproces_gt(gt_mask)
    print("GT SHAPE", gt_mask.shape)
    
    # seg_file_name = os.path.join(seg_pred_dir, t+"_output0")
    seg_file_name = os.path.join(seg_pred_dir, t+"_output.zip")
    seg_mask = torch.load(seg_file_name, map_location=torch.device('cpu'))
    seg_mask = seg_mask.numpy()
    seg_mask = np.squeeze(seg_mask)
    processed_seg_mask, avg, std = preprocess_seg(seg_mask)
    avgs.append(avg)
    stds.append(std)

    '''
    I think what we want is some smarter post processing. 
    Essentially gray is bad if the GT is all black 
    And gray is good if the GT has a tumor. 
    '''

    # Using all dice scores
    SKIP_BLANKS = True
    dice_val = dice_score(processed_seg_mask, processed_gt_mask)
    iou_val = iou(processed_seg_mask, processed_gt_mask)
    if not SKIP_BLANKS:
        dice_vals.append(dice_val)
        iou_vals.append(iou_val)
        if dice_val > 0.001: black_or_not.append(0)
        else: black_or_not.append(1)
        print(f"DICE used: {dice_val}, IOU: {iou_val}")
    else:
        if dice_val > 0.001:
            dice_vals.append(dice_val)
            iou_vals.append(iou_val)
            black_or_not.append(0)
            print(f"DICE used: {dice_val}, IOU: {iou_val}")
        else:
            print(f"DICE skipped: {dice_val}, IOU: {iou_val}")
            black_or_not.append(1)
    # visualize_masks(seg_mask, gt_mask)
    if idx == 1:
        visualize_masks(seg_mask, gt_mask, ground_truth_dir, include_input=True)
        visualize_masks(processed_seg_mask, processed_gt_mask, ground_truth_dir, include_input=True)
        dirs = [
            '../results/dpm_solver++_50steps_ensemble1/',
            '../results/dpm_solver++_50steps_ensemble5/',
            '../results/dpm_solver++_50steps_ensemble10/',
            '../results/dpm_solver++_50steps_ensemble25/',
        ]
        visualize_figure(dirs, processed_gt_mask)
    idx += 1
    


avg_dice = np.mean(dice_vals)
avg_iou = np.mean(iou_vals)
print(f"Avg Dice score: {avg_dice:.4f}")
print(f"Avg IoU: {avg_iou:.4f}")
# print("AVGS", avgs)
# print("STDS", stds)
# print(dice_vals)
# print(black_or_not)