#!/usr/bin/env python

import json
from pathlib import Path
from typing import Tuple

import click
import joblib
import numpy as np
import os
import PIL.Image
import pycocotools.mask
import skvideo.io
from tqdm import tqdm


@click.command()
@click.option("--dataset-path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--segmentation-path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--segmentation-type", type=click.Choice(["json", "png"]))
@click.option("--output-path", type=click.Path(path_type=Path))
@click.option("--num-workers", type=int, default=8)
@click.option("--seed", type=int, default=0)
@click.option("--min-area", type=int, default=64 ** 2 + 1)
@click.option("--min-distance-to-boundary", type=int, default=17)
@click.option("--reject-random", type=float, default=0.0)
def main(**config) -> None:
    examples = [
        child.name for child in config["dataset_path"].iterdir() if child.is_dir()
    ]
   
    joblib.Parallel(n_jobs=config["num_workers"])(
        joblib.delayed(process_example)(config, example)
        for example in tqdm(examples)
    )


def process_example(config: dict, example: str) -> None:
    video = read_video(config, example)
    masks = read_masks(config, example)

    # Initialize the random number generator with a seed depending on the example, so
    # that we get different sequences in different threads.
    random = np.random.default_rng(config["seed"] + int(example))

    for frame_index in range(video.shape[0]):
        object_crops = extract_objects(config, video[frame_index], masks[frame_index], random=random)

        for object_id, crop_image, crop_mask in object_crops:
            object_id = f"sample{example}_object{object_id:03d}"
            frame_id = f"frame{frame_index:03d}"

            os.makedirs(config["output_path"] / object_id, exist_ok=True)

            PIL.Image.fromarray(crop_image).save(
                config["output_path"] / object_id / f"{frame_id}.image.png"
            )
            PIL.Image.fromarray(crop_mask).save(
                config["output_path"] / object_id / f"{frame_id}.mask.png",
                mode="L"
            )


def extract_objects(config: dict, image: np.ndarray, mask: np.ndarray, random: np.random.Generator):
    for object_id in np.unique(mask):
        if object_id <= 1:
            continue

        object_mask = (mask == object_id)
        bounding_box = get_bounding_box(object_mask)

        if reject_bounding_box(bounding_box, config["min_area"], config["min_distance_to_boundary"]):
            continue

        if reject_random(random, config["reject_random"]):
            continue

        crop_image, crop_mask = crop(image, mask, object_mask, bounding_box)

        yield object_id, crop_image, crop_mask


def reject_random(random: np.random.Generator, probability: float, **kwargs) -> bool:
    return random.random() < probability


def reject_bounding_box(bounding_box: Tuple[int],
                        min_area: int,
                        min_distance_to_boundary: int,
                        **kwargs) -> bool:
    left, top, right, bottom = bounding_box
    height = bottom - top
    width = right - left

    return height * width < min_area or \
           top < min_distance_to_boundary or \
           bottom > 320 - min_distance_to_boundary or \
           left < min_distance_to_boundary or \
           right > 480 - min_distance_to_boundary


def get_bounding_box(binary_mask: np.ndarray) -> np.ndarray:
    axis_0 = binary_mask.any(1, keepdims=True)
    axis_1 = binary_mask.any(0, keepdims=True)
    
    top, bottom = np.nonzero(axis_0.flatten())[0][[0, -1]]
    left, right = np.nonzero(axis_1.flatten())[0][[0, -1]]    

    return left, top, right + 1, bottom + 1


def crop(image: np.ndarray,
         mask: np.ndarray,
         object_mask: np.ndarray,
         bounding_box: Tuple[int]):
    left, top, right, bottom = bounding_box

    crop_image = image[top:bottom, left:right]
    
    crop_mask = mask[top:bottom, left:right]
    crop_mask = (crop_mask > 1).astype(np.uint8)
    
    crop_object_mask = object_mask[top:bottom, left:right]
    crop_mask[crop_object_mask] = 2

    return crop_image, crop_mask


def read_video(config: dict, example: str) -> np.ndarray:
    video_path = config["dataset_path"] / example / "video.mp4"
    return skvideo.io.vread(str(video_path))


def read_masks(config: dict, example: str) -> np.ndarray:
    segmentation_type = config.get("segmentation_type", "json")
    sample_path = config["segmentation_path"] / example

    if segmentation_type == "json":
        return read_masks_from_json(sample_path / "objects.json")
    elif segmentation_type == "png":
        return read_masks_from_png(sample_path)
    
    raise ValueError(f"Invalid segmentation type: {segmentation_type}")


def read_masks_from_png(path: Path) -> np.ndarray:
    num_frames = 128
    height = 320
    width = 480
    masks = np.zeros((num_frames, height, width), dtype=np.uint8)

    for frame_index in range(num_frames):
        frame_mask = pil_loader(str(path / f"{frame_index:03d}.png"), mode="L")
        masks[frame_index] = np.array(frame_mask)

    return masks


def pil_loader(path: str, mode: str = "RGB"):
    """open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835).
    Reference: https://pytorch.org/vision/0.8/_modules/torchvision/datasets/folder.html#DatasetFolder
    """
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert(mode)


def read_masks_from_json(path: Path) -> np.ndarray:
    """Returns the object masks from an `objects.json` file."""
    with open(path, 'r') as objects_file:
        objects = json.load(objects_file)

    num_frames = 128
    height = 320
    width = 480
    masks = np.zeros((num_frames, height, width), dtype=np.uint8)

    for object_index, object_ in enumerate(objects):
        for frame_index, encoded_mask in enumerate(object_["masks"]):
            mask = pycocotools.mask.decode(encoded_mask)

            # Set the pixels in the ground truth segmentation map belonging to the
            # current object to the respective object label. Objects are labelled
            # starting with 1 (hence object_index + 1), so that label 0 is used for
            # the background.
            # REVIEW I think this is only needed for the ground truth data?
            masks[frame_index][mask > 0] = object_index + 1

    return masks


if __name__ == "__main__":
    main()
