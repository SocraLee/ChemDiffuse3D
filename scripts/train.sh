#!/bin/bash
# ChemDiffuse3D Training Script
# Requires: HuggingFace Accelerate (configured via `accelerate config`)

# ==========================================
# Single-task training (e.g., 3D Super-Resolution 4x)
# ==========================================
accelerate launch \
    --gpu_ids 0,1 \
    --num_processes 2 \
    chemdiffuse3d/train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --proj-coeff=0 \
    --output-dir="./outputs" \
    --exp-name="3dsr4z_experiment" \
    --resume-step 0 \
    --sampling-steps 5000 \
    --checkpointing-steps 10000 \
    --max-train-steps 400000 \
    --gradient-accumulation-steps 5 \
    --backbone_args_json ./configs/backbone_config.json \
    --task_configs_json ./configs/3dsr4z_config.json \
    --conditioning_args_json ./configs/encoder_config.json \
    --project-name "ChemDiffuse3D"

# ==========================================
# With REPA (REPresentation Alignment) loss
# ==========================================
# accelerate launch \
#     --gpu_ids 0,1 \
#     --num_processes 2 \
#     chemdiffuse3d/train.py \
#     --report-to="wandb" \
#     --allow-tf32 \
#     --mixed-precision="fp16" \
#     --seed=0 \
#     --path-type="linear" \
#     --prediction="v" \
#     --weighting="uniform" \
#     --proj-coeff=0.5 \
#     --z-types dinov2 \
#     --z-weights 1.0 \
#     --output-dir="./outputs" \
#     --exp-name="3dsr4z_repa" \
#     --resume-step 0 \
#     --sampling-steps 5000 \
#     --checkpointing-steps 10000 \
#     --max-train-steps 400000 \
#     --gradient-accumulation-steps 5 \
#     --backbone_args_json ./configs/backbone_config.json \
#     --task_configs_json ./configs/3dsr4z_config.json \
#     --conditioning_args_json ./configs/encoder_config.json \
#     --project-name "ChemDiffuse3D_REPA"

# ==========================================
# Multi-task joint training
# ==========================================
# accelerate launch \
#     --gpu_ids 0,1,2,3 \
#     --num_processes 4 \
#     chemdiffuse3d/train.py \
#     --report-to="wandb" \
#     --allow-tf32 \
#     --mixed-precision="fp16" \
#     --seed=0 \
#     --path-type="linear" \
#     --prediction="v" \
#     --weighting="uniform" \
#     --proj-coeff=0 \
#     --output-dir="./outputs" \
#     --exp-name="joint_training" \
#     --resume-step 0 \
#     --sampling-steps 5000 \
#     --checkpointing-steps 10000 \
#     --max-train-steps 400000 \
#     --gradient-accumulation-steps 5 \
#     --backbone_args_json ./configs/backbone_config.json \
#     --task_configs_json ./configs/task_config.json \
#     --conditioning_args_json ./configs/encoder_config.json \
#     --project-name "ChemDiffuse3D_Joint"
