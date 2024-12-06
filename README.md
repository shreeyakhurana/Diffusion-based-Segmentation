# Efficient Diffusion Sampling for Semantic Segmentation

We forked and updated this repoistory from the imoplementation of [Diffusion Models for Implicit Image Segmentation Ensembles](https://arxiv.org/abs/2112.03145) by Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe Valmaggia, and Philippe C. Cattin.

## Running out code

## Data

For our project, we used the [BRATS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html). The dataset has 3D slices of each MRI scan. Data preprocessing required slicing these into 2D slices, ignoring the first 80 and last 26 slices, and saving the rest in the structure shown below. 
The 2D slices need to be stored in the following structure:

```
data
└───training
│   └───slice0001
│       │   t1.nii.gz
│       │   t2.nii.gz
│       │   flair.nii.gz
│       │   t1ce.nii.gz
│       │   seg.nii.gz
│   └───slice0002
│       │  ...
└───testing
│   └───slice1000
│       │   t1.nii.gz
│       │   t2.nii.gz
│       │   flair.nii.gz
│       │   t1ce.nii.gz
│   └───slice1001
│       │  ...

```

## Replication of our Training and Eval

We set most flags in the same way as our seminal work. 

To run the traning that we ran, you can run: 

```
python3 scripts/segmentation_train.py --data_dir ./data/training --lr 1e-4 --batch_size 10 --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
```

We ran this on an AWS EC2 instance for 300K iterations, which required >50 GPU hours. 

For sampling an ensemble of 5 segmentation masks with the baseline DDPM:
```
python scripts/segmentation_sample.py  --data_dir ./data/testing  --model_path ./results/savedmodel.pt --num_ensemble=5 --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
```

For sampling using DDIM: 
```
python scripts/segmentation_sample.py  --data_dir ./data/testing --model_path ./model_checkpoints/savedmodel.pt --num_ensemble=1 --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --dpm_solver True --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
```

For sampling using DPM-Solver++ with 50 steps and ensemble of 5: 
```
python scripts/segmentation_sample.py  --data_dir ./data/testing  --model_path ./results/savedmodel.pt --num_ensemble=5 --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 50 --dpm_solver True--noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
```

#### Citation of forked work

```
@misc{wolleb2021diffusion,
      title={Diffusion Models for Implicit Image Segmentation Ensembles}, 
      author={Julia Wolleb and Robin Sandkühler and Florentin Bieder and Philippe Valmaggia and Philippe C. Cattin},
      year={2021},
      eprint={2112.03145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
