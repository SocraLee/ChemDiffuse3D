"""
ChemDiffuse3D Training Script.

Multi-task joint training with HuggingFace Accelerate for distributed training.
Supports REPA (REPresentation Alignment) loss and EMA model averaging.

Usage:
    accelerate launch train.py \
        --output-dir ./outputs \
        --exp-name my_experiment \
        --backbone_args_json ./configs/backbone_config.json \
        --conditioning_args_json ./configs/encoder_config.json \
        --task_configs_json ./configs/task_config.json
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.library.impl_abstract.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="xformers.*")

import gc
import argparse
import copy
import logging
import os
import wandb
import math
import json
from pathlib import Path
from tqdm.auto import tqdm

from model.model import JointModel
from model.utils import array2grid, sample_posterior, create_logger, requires_grad
from data.dataset import H5Dataset, h5_worker_init_fn, KeyBasedPretrainDataset
from data.dataloader import MultiTaskDataLoader
from loss import SILoss

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.data import Subset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers.models import AutoencoderKL

logger = get_logger(__name__)


def generate_samples(model, vae, latents_pure_noise, cond_input, latents_scale, latents_bias, args):
    """Generate samples using the Euler-Maruyama SDE sampler."""
    from samplers import euler_maruyama_sampler
    samples = euler_maruyama_sampler(
        model,
        latents_pure_noise,
        cond_input,
        num_steps=50,
        path_type=args.path_type,
        cfg_scale=args.cfg_scale,
        guidance_low=0.,
        guidance_high=1.,
        heun=False,
    ).to(torch.float32)

    B, D, C, H, W = samples.shape
    samples = samples.reshape(B * D, C, H, W)
    samples = vae.decode((samples - latents_bias) / latents_scale).sample
    samples = (samples + 1) / 2.
    return samples.clamp(0, 1)


def evaluate_model(model, ema, vae, dataloader, task_id, args, accelerator):
    """Run evaluation on a single task's dataloader."""
    model.eval()
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(accelerator.device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(accelerator.device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3)).to(accelerator.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)

    vis_samples = None
    vis_gts = None

    with ema.average_parameters():
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                hr_image, dino_lr, vae_hr_x, _, lr_image = batch

                vae_hr_x = sample_posterior(vae_hr_x, latents_scale, latents_bias)
                latents_noise = torch.randn_like(vae_hr_x).to(accelerator.device)

                cond_input = (dino_lr, lr_image, task_id)
                samples = generate_samples(
                    model, vae, latents_noise, cond_input,
                    latents_scale, latents_bias, args
                )

                pred_gray = samples[:, 0:1]
                B, D, C, H, W = hr_image.shape
                gt_gray = hr_image.reshape(B * D, C, H, W)

                if not torch.isnan(pred_gray).any():
                    psnr_metric.update(pred_gray, gt_gray)
                    ssim_metric.update(pred_gray, gt_gray)
                else:
                    logger.warning("NaN in prediction")

                if i == 0:
                    vis_samples = pred_gray
                    vis_gts = gt_gray

    eval_psnr = psnr_metric.compute()
    eval_ssim = ssim_metric.compute()

    if vis_samples is not None and vis_samples.shape[0] > 9:
        indices = torch.randperm(vis_samples.shape[0])[:9]
        vis_samples = vis_samples[indices]
        vis_gts = vis_gts[indices]

    psnr_metric.reset()
    ssim_metric.reset()
    return eval_psnr, eval_ssim, vis_samples, vis_gts


def main(args):
    # --- Setup Accelerator ---
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    checkpoint_dir = f"{save_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator.is_main_process:
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # --- Load Configs ---
    with open(args.task_configs_json, 'r') as f:
        task_configs = json.load(f)
    with open(args.backbone_args_json, 'r') as f:
        backbone_args = json.load(f)
    with open(args.conditioning_args_json, 'r') as f:
        conditioning_args = json.load(f)
    common_z_types = args.z_types

    if accelerator.is_main_process:
        logger.info(f"Loaded {len(task_configs)} tasks: {list(task_configs.keys())}")
        logger.info(f"Loaded backbone args: {backbone_args}")

    # --- Build Model ---
    latent_size = args.resolution // 8
    backbone_args["input_size"] = latent_size
    model = JointModel(
        backbone_args=backbone_args,
        conditioning_args=conditioning_args,
        common_z_types=common_z_types,
        task_configs=task_configs,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    z_weight_dict = {}
    if len(args.z_weights) == len(common_z_types):
        for i, z_type in enumerate(common_z_types):
            z_weight_dict[z_type] = args.z_weights[i]
    else:
        z_weight_dict = {'dinov2': 0.15, 'sam': 0.8, 'clip': 0.05}

    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        accelerator=accelerator,
        weighting=args.weighting,
        z_alignment_weights=z_weight_dict,
        z_types_in_model_output=common_z_types
    )

    if accelerator.is_main_process:
        logger.info(f"JointModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Backbone Parameters: {sum(p.numel() for p in model.backbone.parameters()):,}")
        logger.info(f"Conditioning Module Parameters: {sum(p.numel() for p in model.lr_condition_modules.parameters()):,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- Data ---
    train_dataloaders_dict = {}
    dev_dataloaders_dict = {}
    slice_loader_dict = {}
    sampling_weights = []
    for task_id, config in task_configs.items():
        sampling_weights.append(config['sampling_weight'])
        if "pretrain" not in task_id:
            train_dataset = H5Dataset(
                config['train_data_dir'],
                z_types=common_z_types,
                if_train=True
            )
            if accelerator.is_main_process:
                logger.info(f"{len(train_dataset)} samples for task '{task_id}'")
            train_dataloaders_dict[task_id] = DataLoader(
                train_dataset,
                batch_size=config['train_batch_size'],
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                worker_init_fn=h5_worker_init_fn,
            )
            dev_dataset = H5Dataset(
                config['dev_data_dir'],
                z_types=common_z_types,
                if_train=False
            )
            dev_dataloaders_dict[task_id] = DataLoader(
                dev_dataset,
                batch_size=config['eval_batch_size'],
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            if accelerator.is_main_process:
                logger.info(f"{len(dev_dataset)} dev samples for task '{task_id}'")

            subset_indices = range(4)
            subset_dataset = Subset(dev_dataset, subset_indices)
            slice_loader_dict[task_id] = DataLoader(
                subset_dataset,
                batch_size=config['eval_batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                drop_last=False
            )
        else:
            train_dataset = KeyBasedPretrainDataset(
                config['train_data_dir'],
                key=config['key'],
                z_types=common_z_types,
            )
            if accelerator.is_main_process:
                logger.info(f"{len(train_dataset)} samples for pretrain task '{task_id}'")
            train_dataloaders_dict[task_id] = DataLoader(
                train_dataset,
                batch_size=config['train_batch_size'],
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                worker_init_fn=h5_worker_init_fn,
            )

    # --- Prepare with Accelerate ---
    prepared_train_loaders = {}
    for task_id, loader in train_dataloaders_dict.items():
        prepared_train_loaders[task_id] = accelerator.prepare(loader)
    train_dataloader = MultiTaskDataLoader(prepared_train_loaders, sampling_weights=sampling_weights)

    prepared_dev_loaders = {}
    for task_id, loader in dev_dataloaders_dict.items():
        prepared_dev_loaders[task_id] = accelerator.prepare(loader)
    dev_dataloaders_dict = prepared_dev_loaders
    prepared_slice_loaders = {}
    for task_id, loader in slice_loader_dict.items():
        prepared_slice_loaders[task_id] = accelerator.prepare(loader)
    slice_loader_dict = prepared_slice_loaders

    global_step = 0

    # --- Resume from checkpoint ---
    if args.resume_step > 0:
        model, optimizer = accelerator.prepare(model, optimizer)
        ckpt_path = f"{save_dir}/checkpoints/acc_step_{args.resume_step:07d}"
        accelerator.load_state(ckpt_path)
        global_step = args.resume_step
        if accelerator.is_main_process:
            logger.info(f"Resumed from checkpoint: {ckpt_path}")
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
        ema.to(device)
    else:
        if args.pretrain_ckpt is not None:
            state_dict = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True)
            # strict=False: legacy ckpts always contain both upsampler and matched-depth
            # decoder subtrees; the new model may build only one of them. Any unexpected
            # key must come from those two skipped subtrees, and missing keys must be empty.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            allowed_unexpected_prefixes = (
                "lr_condition_modules.decoder.upsampler.",
                "lr_condition_modules.decoder.decoder.",
            )
            bad_unexpected = [k for k in unexpected if not k.startswith(allowed_unexpected_prefixes)]
            if missing or bad_unexpected:
                raise RuntimeError(
                    f"pretrain_ckpt load mismatch.\n  missing={missing}\n  unexpected (unaccounted)={bad_unexpected}"
                )
            if accelerator.is_main_process:
                logger.info(f"Initialized model weights from {args.pretrain_ckpt} "
                            f"(skipped {len(unexpected)} keys from unbuilt cond-decoder subtrees)")
        model, optimizer = accelerator.prepare(model, optimizer)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
        ema.to(device)
        
    if args.resume_step > 0:
        ema_path = f"{save_dir}/checkpoints/acc_step_{args.resume_step:07d}/ema.pt"
        if os.path.exists(ema_path):
            ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=True))
            if accelerator.is_main_process:
                logger.info(f"Resumed EMA from: {ema_path}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"EMA checkpoint not found at {ema_path}, starting EMA from scratch.")

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=args.project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}}
        )

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    accelerator.wait_for_everyone()

    # --- Training Loop ---
    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(accelerator.device)
    latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(accelerator.device)

    for epoch in range(args.epochs):
        model.train()
        for task_id, batch in train_dataloader:
            hr_image, dino_lr_y, vae_hr_x, hr_embeddings, lr = batch
            vae_hr_x = sample_posterior(vae_hr_x, latents_scale=latents_scale, latents_bias=latents_bias)
            zs = {z_type: hr_embeddings[z_type] for z_type in common_z_types}

            with accelerator.accumulate(model):
                feature_alignment = (args.proj_coeff != 0)
                model_kwargs = dict(
                    y=dino_lr_y,
                    lr_image=lr,
                    feature_alignment=feature_alignment
                )
                loss, proj_loss = loss_fn(model, vae_hr_x, model_kwargs, zs=zs)

                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()

                loss = loss_mean + proj_loss_mean * args.proj_coeff

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm = grad_norm.detach()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    ema.update()

            # Logging and saving
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                loss_mean_detached = loss_mean.detach()
                proj_loss_mean_detached = proj_loss_mean.detach()
                logs = {
                    "train/loss": accelerator.gather(loss_mean_detached).mean().item(),
                    "train/proj_loss": accelerator.gather(proj_loss_mean_detached).mean().item(),
                    "train/grad_norm": accelerator.gather(grad_norm).mean().item(),
                    "epoch": epoch
                }
                accelerator.log(logs, step=global_step)

                # Checkpoint
                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path = f"{checkpoint_dir}/acc_step_{global_step:07d}"
                        accelerator.save_state(save_path)
                        ema.to("cpu")
                        torch.save(ema.state_dict(), f"{save_path}/ema.pt")
                        with ema.average_parameters():
                            torch.save(
                                {k: v.cpu() for k, v in model.module.state_dict().items()}
                                if hasattr(model, 'module')
                                else {k: v.cpu() for k, v in model.state_dict().items()},
                                f"{save_path}/ema_named.pt"
                            )
                        ema.to(device)
                        logger.info(f"Saved state to {save_path}")

                # Test run at step 5
                if global_step == 5:
                    if accelerator.is_main_process:
                        tqdm.write("Test run of evaluation...")
                    for eval_task_id, dev_loader in slice_loader_dict.items():
                        eval_psnr, eval_ssim, vis_samples, gt_samples = evaluate_model(
                            model, ema, vae, dev_loader, eval_task_id, args, accelerator
                        )
                        del eval_psnr, eval_ssim, vis_samples, gt_samples

                # Periodic evaluation
                if (global_step % args.sampling_steps == 0 and global_step > 0):
                    if accelerator.is_main_process:
                        logger.info("Running evaluation...")
                    model.eval()
                    all_eval_metrics = {}

                    for eval_task_id, dev_loader in dev_dataloaders_dict.items():
                        eval_psnr, eval_ssim, vis_samples, gt_samples = evaluate_model(
                            model, ema, vae, dev_loader, eval_task_id, args, accelerator
                        )

                        if accelerator.is_main_process:
                            all_eval_metrics[f"eval/{eval_task_id}/psnr"] = eval_psnr
                            all_eval_metrics[f"eval/{eval_task_id}/ssim"] = eval_ssim
                            if eval_task_id == list(dev_dataloaders_dict.keys())[0]:
                                all_eval_metrics["samples"] = wandb.Image(array2grid(vis_samples))
                                all_eval_metrics["gt_samples"] = wandb.Image(array2grid(gt_samples))

                    if accelerator.is_main_process:
                        accelerator.log(all_eval_metrics, step=global_step)

                    model.train()

            gc.collect()
            del batch, hr_image, dino_lr_y, vae_hr_x, hr_embeddings, lr
            del proj_loss, loss

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="ChemDiffuse3D Training")

    # Logging
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")

    parser.add_argument("--sampling-steps", type=int, default=10000,
                        help="Steps interval for evaluation.")
    parser.add_argument("--checkpointing-steps", type=int, default=10000,
                        help="Steps interval for checkpointing.")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step to resume training from.")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Path to a plain model state_dict (.pt).")

    # Config Files
    parser.add_argument("--backbone_args_json", type=str, required=True,
                        help="Path to JSON file with SiT_3D_Backbone args.")
    parser.add_argument("--conditioning_args_json", type=str, required=True,
                        help="Path to JSON file defining encoder params.")
    parser.add_argument("--task_configs_json", type=str, required=True,
                        help="Path to JSON file defining all tasks and data paths.")

    # Model & Training
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=9999)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    # Optimizer
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    # System
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true")

    # Diffusion & Loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])
    parser.add_argument("--cfg-prob", type=float, default=0.)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--project-name", type=str, default='ChemDiffuse3D')
    parser.add_argument("--proj-coeff", type=float, default=0)
    parser.add_argument("--weighting", default="uniform", type=str)

    # REPA (optional)
    parser.add_argument('--z-types', nargs='+', default=[])
    parser.add_argument('--z-weights', nargs='+', type=float, default=[])

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
