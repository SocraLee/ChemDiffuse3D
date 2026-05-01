#!/bin/bash
# MicroDiffuse3D Post-Diffusion Decoder Training

# ==========================================
# Adapted Decoder for 3D Super-Resolution
# ==========================================
python microdiffuse3d/train_decoder.py \
    --task_configs_json configs/3dsr4z_config.json \
    --project-name MicroDiffuse3D_Decoder \
    --exp-name 3dsr4z_adapted \
    --decoder-type adapted \
    --save-dir ./outputs/decoder \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 50 \
    --val-freq 2 \
    --num-workers 8


# ==========================================
# Adapted Decoder for 3D Denoising
# ==========================================
# python microdiffuse3d/train_decoder.py \
#     --task_configs_json configs/3ddenoise_config.json \
#     --project-name MicroDiffuse3D_Decoder \
#     --exp-name denoise_adapted \
#     --decoder-type adapted \
#     --save-dir ./outputs/decoder \
#     --batch-size 4 \
#     --lr 1e-4 \
#     --epochs 50 \
#     --val-freq 2 \
#     --num-workers 8
