import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Tuple

import click
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
import torch.optim
import torch.utils.data
import torch.utils.tensorboard
import torchvision.utils
from tqdm import tqdm
import yaml

from common_fate_object_learning import datasets
from common_fate_object_learning import transforms
from common_fate_object_learning import metrics
from common_fate_object_learning import models
from common_fate_object_learning import visualization


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")



@click.group()
def cli():
    pass


@cli.command()
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              default="configs/dev.yaml")
@click.option("--output-path", type=click.Path(exists=False), default=None)
def train(config: str, output_path: str) -> None:
    config = load_config(config)

    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = Path(config["output_path"])
    
    os.makedirs(output_path)

    set_global_seed(config.get("seed", 0))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(config["model"])
    model.cuda()

    summary_writer = torch.utils.tensorboard.SummaryWriter(log_dir=output_path)

    training_dataloader = build_training_dataloader(config["dataloaders"]["training"])
    hallucinate = config["dataloaders"]["training"].get("hallucinate", False)
    visualize_training_data(training_dataloader, summary_writer, hallucinate=hallucinate)

    validation_dataloader = build_validation_dataloader(config["dataloaders"]["validation"])

    optimizer, scheduler = \
        build_optimizer_and_scheduler(model, config["optimizer"], config["scheduler"])

    LOGGER.info("Training")
    num_epochs = config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        train_one_epoch(model, training_dataloader, optimizer, epoch, num_epochs, summary_writer, hallucinate=hallucinate)

        scheduler.step()

        if (epoch + 1) % config["training"]["validation_frequency"] == 0:
            evaluate_and_log(validation_dataloader, model, output_path, epoch, summary_writer)
            visualize_validation_output(validation_dataloader, model, summary_writer, epoch)

        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(state_dict, output_path / "checkpoint_last.pth")
        if (epoch + 1) % config["training"]["keep_checkpoint_frequency"] == 0:
            torch.save(state_dict, output_path / f"checkpoint_epoch{epoch}.pth")
    
    LOGGER.info("Final evaluation")
    for seed in config["final_evaluation"]["seeds"]:
        set_global_seed(seed)
        results = evaluate_model(validation_dataloader, model)
        results.to_csv(output_path / f"results_final{seed}.csv", index=False)


@cli.command()
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False),
              default="configs/default.yaml")
@click.option("--output-path", type=click.Path(exists=False))
@click.option("--checkpoint-path")
def sample(config: str, output_path: str, checkpoint_path: str) -> None:
    config = load_config(config)

    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    set_global_seed(config.get("seed", 0))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(config["model"])
    model.cuda()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    sample = model.sample(144, device="cuda")
    sample_image = sample[:, :3]
    sample_mask = sample[:, 3:]
    sample_image = visualization.normalize(sample_image)
    sample_image_masked = torch.clamp((sample_image + (sample_mask < 0.0).type(torch.float32)), 0.0, 1.0)

    torchvision.utils.save_image(sample_image_masked, output_path, nrow=12, normalize=False)


def load_config(path: str) -> Dict:
    with open(path, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config: Dict) -> models.BetaVAE:
    LOGGER.info("Building model")

    type_ = config["type"]
    kwargs = {key: value for key, value in config.items() if key != "type"}

    if type_ == "BetaVAE":
        return models.BetaVAE(**kwargs)
    
    if type_ == "DoubleDecoderVAE":
        return models.DoubleDecoderVAE(**kwargs)

    raise ValueError(f"Unknown model type: {type_}")


def build_training_dataloader(config: Dict) -> torch.utils.data.DataLoader:
    LOGGER.info("Building training data loader")

    transform = transforms.ComposedTransform([
            transforms.Normalize,
            transforms.CutoutAugmentation
        ],
        num_distractors = config["num_distractors"],
        fill_mode = 0.,
        image_mean = [.5,.5,.5],
        image_std = [.5,.5,.5]
    )

    dataset = datasets.ObjectDataset(
        config["data_path"],
        config["num_frames_per_object"],
        size=config["image_size"],
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True
    )
    
    return dataloader


def build_validation_dataloader(config: Dict) -> torch.utils.data.DataLoader:
    LOGGER.info("Building validation data loader")

    transform = transforms.Normalize(image_mean=[.5, .5, .5], image_std=[.5, .5, .5])

    dataset = datasets.ObjectDatasetValidation(
        config["data_path"],
        sample_size=config["image_size"],
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=False
    )
    
    return dataloader


def build_optimizer_and_scheduler(model: torch.nn.Module,
                                  optimizer_config: Dict,
                                  scheduler_config: Dict) -> Tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)

    scheduler_type = scheduler_config.get("type", "ExponentialLR").lower()
    scheduler_config = {key: value for key, value in scheduler_config.items() if key != "type"}

    if scheduler_type == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
    elif scheduler_type == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return optimizer, scheduler


def hallucinate_occlusions(images, masks):
    batch_dims = images.shape[:-3]
    images = images.flatten(end_dim=-4)
    masks = masks.flatten(end_dim=-3)
    num_samples, channels, height, width = images.shape

    displacement_y = ((torch.rand(num_samples) - 0.5) * height).type(torch.long)
    displacement_x = ((torch.rand(num_samples) - 0.5) * width).type(torch.long)

    occlusion_left = torch.max(torch.tensor(0), displacement_x)
    occlusion_top = torch.max(torch.tensor(0), displacement_y)
    occlusion_right = torch.min(torch.tensor(width), width + displacement_x)
    occlusion_bottom = torch.min(torch.tensor(height), height + displacement_y)

    occluder_left = torch.max(torch.tensor(0), -displacement_x)
    occluder_top = torch.max(torch.tensor(0), -displacement_y)
    occluder_right = torch.min(torch.tensor(width), width - displacement_x)
    occluder_bottom = torch.min(torch.tensor(width), height - displacement_y)

    occluded_images = images.clone()

    for sample_index in range(images.shape[0]):
        occlusion_index = (sample_index + batch_dims[1]) % images.shape[0]
        
        occlusion_image = images[occlusion_index, :, occluder_top[sample_index]:occluder_bottom[sample_index], occluder_left[sample_index]:occluder_right[sample_index]]
        occlusion_mask = masks[occlusion_index, occluder_top[sample_index]:occluder_bottom[sample_index], occluder_left[sample_index]:occluder_right[sample_index]]

        occluded_image = occluded_images[sample_index, :, occlusion_top[sample_index]:occlusion_bottom[sample_index], occlusion_left[sample_index]:occlusion_right[sample_index]]
        occluded_image[:, occlusion_mask > 0] = occlusion_image[:, occlusion_mask > 0]

    return occluded_images.view(*batch_dims, channels, height, width)


def train_one_epoch(model, training_dataloader, optimizer, epoch, num_epochs, summary_writer, hallucinate=False):
    model.train()

    iterator = tqdm(training_dataloader, desc=f"Epoch {epoch}/{num_epochs}")

    for batch_index, (images_clean, images_augmented, masks) in enumerate(iterator):
        images_clean = images_clean.cuda()
        images_augmented = images_augmented.cuda()
        masks = masks.cuda()

        if hallucinate:
            images_augmented = hallucinate_occlusions(images_augmented, masks)

        output = model.forward(images_augmented)

        overall_loss, partial_losses = model.loss(
            *output,
            target_image=images_clean,
            target_mask=masks
        )
        
        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()

        global_step = (epoch * len(training_dataloader) + batch_index) * images_clean.shape[0]
        summary_writer.add_scalar("loss", overall_loss, global_step=global_step)

        for partial_loss_key, partial_loss_value in partial_losses.items():
            summary_writer.add_scalar(f"partial_losses/{partial_loss_key}", partial_loss_value, global_step=global_step)

        iterator.set_postfix_str(f"loss={overall_loss.item():05f}")


def evaluate_model(validation_dataloader, model, threshold=0.0):
    results = []

    for batch in tqdm(validation_dataloader):
        images = batch["image"].cuda()
        target_images = batch["target_image"].cuda() / 2.0 + 0.5
        target_masks = (batch["target_mask"] == 2).cuda()  # 2 = object of interest

        with torch.no_grad():
            prediction = model.reconstruct(images)
            predicted_images = prediction[:, :, :3] / 2.0 + 0.5
            predicted_masks = prediction[:, :, 3] > threshold

            mse_mask = target_masks & predicted_masks
            mse = metrics.masked_mse(predicted_images, target_images, mse_mask).flatten()
            mae = metrics.masked_mae(predicted_images, target_images, mse_mask).flatten()

            intersection = torch.sum(target_masks & predicted_masks, dim=[-3, -2, -1])
            union = torch.sum(target_masks | predicted_masks, dim=[-3, -2, -1])

            sample_iou = torch.zeros_like(union, dtype=torch.float32)

            iou_mask = union > 0
            sample_iou[iou_mask] = intersection[iou_mask] / union[iou_mask]
            sample_iou = sample_iou.flatten()

            for index in range(mse.shape[0]):
                results.append({
                    "sample": batch["sample_id"][index].item(),
                    "object": batch["object_id"][index].item(),
                    "frame": batch["frame_id"][index].item(),
                    "mse": mse[index].item(),
                    "mae": mae[index].item(),
                    "iou": sample_iou[index].item()
                })
        
    return pd.DataFrame(results)


def evaluate_and_log(validation_dataloader, model, output_path, epoch, summary_writer, threshold=0.0):
    model.eval()

    results = evaluate_model(validation_dataloader, model, threshold)

    mean_iou = results["iou"].mean()
    mean_mae = results["mae"].mean()
    mean_mse = results["mse"].mean()

    results.to_csv(output_path / f"results_epoch{epoch}.csv", index=False)

    summary_writer.add_scalar("validation/iou", mean_iou, global_step=epoch)
    summary_writer.add_scalar("validation/mae", mean_mae, global_step=epoch)
    summary_writer.add_scalar("validation/mse", mean_mse, global_step=epoch)


def visualize_training_data(training_dataloader, summary_writer, hallucinate=False):
    images_clean, images_augmented, masks = next(iter(training_dataloader))

    if hallucinate:
        images_augmented = hallucinate_occlusions(images_augmented, masks)

    # Merge batch and time dimensions
    images_clean = images_clean.view(-1, *images_clean.shape[-3:])
    images_augmented = images_augmented.view(-1, *images_augmented.shape[-3:])

    images_clean = torchvision.utils.make_grid(images_clean, normalize=True, nrow=12)
    images_augmented = torchvision.utils.make_grid(images_augmented, normalize=True, nrow=12)

    masks = masks.view(-1, *masks.shape[-2:]).unsqueeze(1).type(torch.float)
    masks = torchvision.utils.make_grid(masks, normalize=True, nrow=12)

    summary_writer.add_image("training/images_clean", images_clean)
    summary_writer.add_image("training/images_augmented", images_augmented)
    summary_writer.add_image("training/masks", masks)


def visualize_validation_output(validation_dataloader, model, summary_writer, global_step):
    batch = next(iter(validation_dataloader))
    images = batch["image"].cuda()
    target_images = batch["target_image"]
    target_masks = batch["target_mask"].unsqueeze(-3)  # unsqueeze channel dim

    with torch.no_grad():
        reconstruction = model.reconstruct(images)
        reconstructed_image = reconstruction[:, :, :3].cpu()
        reconstructed_mask = reconstruction[:, :, 3:].cpu()

        images = images.view(-1, *images.shape[-3:])
        images = torchvision.utils.make_grid(images, normalize=True, nrow=12)
        summary_writer.add_image("validation/input_image", images, global_step=global_step)

        target_images = target_images.view(-1, *target_images.shape[-3:])
        target_masks = target_masks.view(-1, *target_masks.shape[-3:]).type(torch.float)

        target_images = visualization.normalize(target_images)
        target_images = torch.clamp((target_images + (target_masks <= 0).type(torch.float32)), 0.0, 1.0)

        target_images = torchvision.utils.make_grid(target_images, normalize=False, nrow=12)
        target_masks = torchvision.utils.make_grid(target_masks, normalize=True, nrow=12)

        summary_writer.add_image("validation/target_object", target_images, global_step=global_step)
        summary_writer.add_image("validation/target_mask", target_masks, global_step=global_step)

        reconstructed_image = reconstructed_image.view(-1, *reconstructed_image.shape[-3:])
        reconstructed_mask = reconstructed_mask.view(-1, *reconstructed_mask.shape[-3:])

        reconstructed_image = visualization.normalize(reconstructed_image)
        reconstructed_image_masked = torch.clamp((reconstructed_image + (reconstructed_mask < 0.0).type(torch.float32)), 0.0, 1.0)

        reconstructed_image = torchvision.utils.make_grid(reconstructed_image, normalize=False, nrow=12)
        reconstructed_mask = torchvision.utils.make_grid(reconstructed_mask, normalize=True, nrow=12)
        reconstructed_image_masked = torchvision.utils.make_grid(reconstructed_image_masked, normalize=False, nrow=12)

        summary_writer.add_image("validation/reconstructed_image", reconstructed_image, global_step=global_step)
        summary_writer.add_image("validation/reconstructed_mask", reconstructed_mask, global_step=global_step)
        summary_writer.add_image("validation/reconstructed_object", reconstructed_image_masked, global_step=global_step)


if __name__ == "__main__":
    cli()
