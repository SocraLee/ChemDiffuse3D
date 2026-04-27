"""
ChemDiffuse3D Decoder Training Script.

Trains the post-diffusion adapted decoder that takes VAE latents
and LR volumes as input to produce high-resolution output images.

Usage:
    python train_decoder.py \
        --task_configs_json ./configs/3dsr4z_config.json \
        --exp-name denoise_adapted \
        --decoder-type adapted \
        --save-dir ./outputs/decoder \
        --batch-size 4 \
        --epochs 50
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from data.dataset import H5Dataset
from model.decoder import AdaptedDecoder


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def get_test_path(train_path):
    if "/train/" in train_path:
        return train_path.replace("/train/", "/test/")
    return train_path.replace("train", "test")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    with open(args.task_configs_json, 'r') as f:
        task_configs = json.load(f)

    for task_id, config in task_configs.items():
        print(f"\n{'=' * 50}\nStarting Training for Task: {task_id}\n{'=' * 50}")

        if args.report_to == 'wandb':
            wandb.init(project=args.project_name, name=f"{args.exp_name}_{task_id}", config=vars(args))

        # Data
        train_path = config['train_data_dir']
        val_path = get_test_path(train_path)

        if not os.path.exists(train_path):
            print(f"Warning: Train data not found for {task_id}, skipping.")
            if args.report_to == 'wandb':
                wandb.finish()
            continue

        train_dataset = H5Dataset(train_path, z_types=args.z_types, if_train=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )

        if os.path.exists(val_path):
            val_dataset = H5Dataset(val_path, z_types=args.z_types, if_train=False)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
        else:
            val_dataset, val_loader = None, None

        # Model
        print(f"Initializing {args.decoder_type} decoder...")
        if args.decoder_type == 'adapted':
            model = AdaptedDecoder().to(device)
        else:
            raise ValueError(f"Unknown decoder type: {args.decoder_type}")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")

        if args.report_to == 'wandb':
            wandb.watch(model, log="all", log_freq=100)

        # Optimizer and Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        warmup = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=2)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - 2, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])

        criterion = CharbonnierLoss(eps=1e-3)
        compute_psnr = PeakSignalNoiseRatio(data_range=1.0, dim=[1, 2, 3]).to(device)
        compute_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # Save directory
        task_save_dir = os.path.join(args.save_dir, args.decoder_type, config.get('decoder_folder', task_id))
        os.makedirs(task_save_dir, exist_ok=True)

        # Training Loop
        best_val_psnr = 0.0
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"[{task_id}] Epoch {epoch}/{args.epochs} [Train]")
            for step_idx, batch in enumerate(pbar):
                hr_image, dino_lr_y, vae_hr_x, _, lr_image = batch

                hr_image = hr_image.to(device)
                lr_image = lr_image.to(device)
                vae_hr_x = vae_hr_x.to(device)

                target = hr_image.permute(0, 2, 1, 3, 4)
                latent = vae_hr_x[:, :, :4].permute(0, 2, 1, 3, 4)

                output = model(lr_image, latent)

                loss = criterion(output, target)
                scaled_loss = loss / args.grad_accum_steps
                scaled_loss.backward()

                if (step_idx + 1) % args.grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * lr_image.size(0)
                global_step += 1

                pbar.set_postfix({"loss": loss.item()})
                if args.report_to == 'wandb':
                    wandb.log({"train/loss_step": loss.item(), "global_step": global_step})

            avg_train_loss = train_loss / len(train_dataset)
            if args.report_to == 'wandb':
                wandb.log({"train/loss_epoch": avg_train_loss, "epoch": epoch})

            # Validation
            if epoch % args.val_freq == 0 and val_loader is not None:
                model.eval()
                val_loss = 0.0
                compute_psnr.reset()
                compute_ssim.reset()

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"[{task_id}] Epoch {epoch}/{args.epochs} [Val]")
                    for batch in val_pbar:
                        hr_image, dino_lr_y, vae_hr_x, _, lr_image = batch
                        hr_image = hr_image.to(device)
                        lr_image = lr_image.to(device)
                        vae_hr_x = vae_hr_x.to(device)

                        target = hr_image.permute(0, 2, 1, 3, 4)
                        latent = vae_hr_x[:, :, :4].permute(0, 2, 1, 3, 4)

                        output = model(lr_image, latent)
                        loss = criterion(output, target)
                        val_loss += loss.item() * lr_image.size(0)

                        output_clamped = torch.clamp(output, 0, 1)
                        for i in range(target.shape[0]):
                            compute_psnr.update(
                                output_clamped[i].permute(1, 0, 2, 3),
                                target[i].permute(1, 0, 2, 3)
                            )
                            compute_ssim.update(
                                output_clamped[i].permute(1, 0, 2, 3),
                                target[i].permute(1, 0, 2, 3)
                            )

                avg_val_loss = val_loss / len(val_dataset)
                val_psnr = compute_psnr.compute().item()
                val_ssim = compute_ssim.compute().item()

                print(f"[{task_id}] Val: Loss={avg_val_loss:.6f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")
                if args.report_to == 'wandb':
                    wandb.log({
                        "val/loss": avg_val_loss,
                        "val/psnr": val_psnr,
                        "val/ssim": val_ssim,
                        "epoch": epoch
                    })

                if val_psnr > best_val_psnr + 0.2:
                    best_val_psnr = val_psnr
                    ckpt_path = os.path.join(task_save_dir, "best_decoder.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved new best checkpoint to {ckpt_path}")

            # Always save latest checkpoint
            torch.save(model.state_dict(), os.path.join(task_save_dir, "latest_decoder.pth"))
            scheduler.step()
            if args.report_to == 'wandb':
                wandb.log({"learning_rate": scheduler.get_last_lr()[0], "epoch": epoch})

        print(f"Task {task_id} Training Complete.")
        if args.report_to == 'wandb':
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChemDiffuse3D Decoder Training")
    parser.add_argument("--task_configs_json", type=str, required=True)
    parser.add_argument("--project-name", type=str, default="ChemDiffuse3D_Decoder")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--decoder-type", type=str, choices=['adapted'], default='adapted')
    parser.add_argument("--save-dir", type=str, required=True, help="Base directory to save decoder checkpoints")
    parser.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "none"])

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument('--z-types', nargs='+', default=[])

    args = parser.parse_args()
    main(args)
