#!/bin/bash
# ChemDiffuse3D Evaluation Script

# ==========================================
# Standard evaluation (VAE decoder)
# ==========================================
CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
    --task_configs_json "./configs/3dsr4z_config.json" \
    --backbone_args_json "./configs/backbone_config.json" \
    --conditioning_args_json "./configs/encoder_config.json" \
    --ckpt_dir "./outputs/3dsr4z_experiment/checkpoints" \
    --resume-steps 40000 \
    --exp-name "3dsr4z_eval_40k" \
    --save-results \
    --batch-size 4 \
    --num-steps 50 \
    --project-name "ChemDiffuse3D" \
    --results-key chemdiffuse3d_output

# ==========================================
# Evaluation with RCAN post-diffusion decoder
# ==========================================
# CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
#     --task_configs_json "./configs/3dsr4z_config.json" \
#     --backbone_args_json "./configs/backbone_config.json" \
#     --conditioning_args_json "./configs/encoder_config.json" \
#     --ckpt_dir "./outputs/3dsr4z_experiment/checkpoints" \
#     --resume-steps 40000 \
#     --exp-name "3dsr4z_eval_rcan" \
#     --save-results \
#     --batch-size 4 \
#     --num-steps 50 \
#     --decoder-type rcan \
#     --decoder-path "./outputs/decoder/rcan/3d_sr/best_decoder.pth" \
#     --results-key chemdiffuse3d_output_rcan

# ==========================================
# Evaluation with Fused decoder
# ==========================================
# CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
#     --task_configs_json "./configs/biotisr_config.json" \
#     --backbone_args_json "./configs/backbone_config.json" \
#     --conditioning_args_json "./configs/encoder_config.json" \
#     --ckpt_dir "./outputs/biotisr_experiment/checkpoints" \
#     --resume-steps 90000 \
#     --exp-name "biotisr_eval_fused" \
#     --save-results \
#     --batch-size 4 \
#     --num-steps 50 \
#     --decoder-type fused \
#     --decoder-path "./outputs/decoder/fused/biotisr/best_decoder.pth" \
#     --results-key chemdiffuse3d_output_fused
