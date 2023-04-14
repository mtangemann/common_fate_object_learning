import abc
from dataclasses import dataclass
from pathlib import Path
import random
import re
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from common_fate_object_learning import io_utils


LOGGER = logging.getLogger(__name__)


BACKGROUND = 0
OCCLUSION = 1
FOREGROUND = 2


class _SampleFrame(abc.ABC):
    @abc.abstractstaticmethod
    def from_path(path: Path, name: str) -> Any:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_valid(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, size: int) -> Any:
        raise NotImplementedError

    def _get_padding(self, image_size: Tuple[int], target_size: int):
        width, height = image_size
    
        target_width = max(height*2, width + width % 2)
        target_height = target_width // 2    
        assert target_height * 2 == target_width
    
        top_pad    = (target_height - height) // 2
        bottom_pad = target_height - height - top_pad
        left_pad   = (target_width - width) // 2
        right_pad  = target_width - width - left_pad

        return left_pad, top_pad, right_pad, bottom_pad


@dataclass
class _SampleFrameTraining(_SampleFrame):
    image_path: Path
    mask_path: Path

    @staticmethod
    def from_path(path: Path, name: str):
        return _SampleFrameTraining(
            path / f"{name}.image.png",
            path / f"{name}.mask.png"
        )

    def is_valid(self) -> bool:
        return self.image_path.is_file() and self.mask_path.is_file()

    def load(self, size: int = 32):
        image = io_utils.pil_loader(self.image_path, mode="RGB")
        mask = io_utils.pil_loader(self.mask_path, mode="L")

        padding = self._get_padding(image.size, size)

        image = TF.pad(image, padding=padding, padding_mode='edge')
        mask = TF.pad(mask, padding=padding, padding_mode='constant', fill=0)

        image = TF.to_tensor(TF.resize(image, size, PIL.Image.NEAREST))
        mask = torch.from_numpy(np.asarray(TF.resize(mask, size, PIL.Image.NEAREST))).long()

        return image, mask


@dataclass
class _SampleFrameValidation(_SampleFrame):
    sample_id: int
    object_id: int
    frame_id: int
    image_path: Path
    mask_path: Path
    target_image_path: Path
    target_mask_path: Path

    @staticmethod
    def from_path(path: Path, name: str):
        match = re.match(r".*sample(\d+)_object(\d+)/frame(\d+).*", str(path / name))
        if not match:
            LOGGER.warning("Warning: Unable to parse sample path: {}", path / name)
            sample_id = None
            object_id = None
            frame_id = None
        else:
            sample_id = int(match.group(1))
            object_id = int(match.group(2))
            frame_id = int(match.group(3))

        return _SampleFrameValidation(
            sample_id,
            object_id,
            frame_id,
            path / f"{name}.image.png",
            path / f"{name}.mask.png",
            path / f"{name}.image-full.png",
            path / f"{name}.mask-full.png",
        )
    
    def is_valid(self):
        return self.image_path.is_file() and \
               self.mask_path.is_file() and \
               self.target_image_path.is_file() and \
               self.target_mask_path.is_file()

    def load(self, size: int = 32):
        image = io_utils.pil_loader(self.image_path, mode="RGB")
        mask = io_utils.pil_loader(self.mask_path, mode="L")
        target_image = io_utils.pil_loader(self.target_image_path, mode="RGB")
        target_mask = io_utils.pil_loader(self.target_mask_path, mode="L")

        padding = self._get_padding(image.size, size)

        image = TF.pad(image, padding=padding, padding_mode='edge')
        image = TF.to_tensor(TF.resize(image, size, PIL.Image.NEAREST))

        mask = TF.pad(mask, padding=padding, padding_mode='constant', fill=0)
        mask = torch.from_numpy(np.asarray(TF.resize(mask, size, PIL.Image.NEAREST))).long()

        target_image = TF.pad(target_image, padding=padding, padding_mode='edge')
        target_image = TF.to_tensor(TF.resize(target_image, size, PIL.Image.NEAREST))

        target_mask = TF.pad(target_mask, padding=padding, padding_mode='constant', fill=0)
        target_mask = torch.from_numpy(np.asarray(TF.resize(target_mask, size, PIL.Image.NEAREST))).long()

        return {
            "sample_id": self.sample_id,
            "object_id": self.object_id,
            "frame_id": self.frame_id,
            "image": image,
            "mask": mask,
            "target_image": target_image,
            "target_mask": target_mask
        }


class ObjectDataset(torch.utils.data.Dataset):
    """Loads object crops with segmentation masks.
    
    This assumes the following file structure:
    ```
    dataset/sample<sample_id>_object<object_id>/frame<frame_id>.image.png
    dataset/sample<sample_id>_object<object_id>/frame<frame_id>.mask.png
    ```

    The `*.image.png` files are the cropped objects as RGB images, the
    `*.masks.png` are grayscale images defining the object masks using the
    following code:
    - 0 = background
    - 1 = other object
    - 2 = object of interest

    The length of this dataset is the number of objects. When retrieving an
    object, this will randomly sample the given number of frames for this object.
    """

    def __init__(self,
                 path: str,
                 num_frames_per_object: int = 1,
                 size: int = 32,
                 transform: Optional[Callable] = None,
                 load_evaluation_data: bool = False,
                 old_format: bool = False) -> None:
        """Initializes the data laoder

        Args:
            path: Path to the dataset.
            num_frames_per_object: Number of frames to load per object.
            size: Return samples resized and cropped to (size, size*2).
            transform: Optional transform to apply to images and masks.
            old_format: If set to true, the old data format is used
                without folders per object. Every (image,mask) pair is then
                interpreted as individual objects.
        """
        self._path = Path(path)
        self._num_frames_per_object = num_frames_per_object
        self.size = size
        self.transform = transform
        self._load_evaluation_data = load_evaluation_data
        self._old_format = old_format

        self._load_samples()

    def _load_samples(self) -> None:
        self._samples = []

        if self._old_format:
            # Old format = No subfolder per object
            # This will interprete every image pair as individual object
            if self._num_frames_per_object <= 1:
                self._samples = self._load_sample_frames(path)
                self._samples = [List[frame] for frame in self._samples]
            return

        for sample_path in self._path.iterdir():
            if sample_path.is_dir():
                frames = self._load_sample_frames(sample_path)

                if len(frames) > self._num_frames_per_object:
                    self._samples.append(frames)

    def _load_sample_frames(self, path: Path) -> List[_SampleFrame]:
        frames = []

        for image_path in path.glob("*.image.png"):
            name = image_path.name[:-len(".image.png")]

            if not self._load_evaluation_data:
                frame = _SampleFrameTraining.from_path(image_path.parent, name)
            else:
                frame = _SampleFrameValidation.from_path(image_path.parent, name)

            if frame.is_valid():
                frames.append(frame)

        return frames

    def __len__(self):
        """Returns the number of samples in this dataset."""
        return len(self._samples)

    def __getitem__(self, index):
        """Returns the sample with the given index."""
        frames = random.sample(self._samples[index], self._num_frames_per_object)
        frames = [frame.load(self.size) for frame in frames]

        if not self._load_evaluation_data:
            images = torch.stack([image for image, _ in frames])
            masks = torch.stack([mask for _, mask in frames])

            if self.transform is not None:
                return self.transform(images, masks)

            return images, masks

        else:
            stacked_frames = {}

            for key in frames[0].keys():
                stacked_frames[key] = torch.stack([
                    frame[key] if isinstance(frame[key], torch.Tensor) else torch.tensor(frame[key])
                    for frame in frames
                ])

            if self.transform is not None:
                stacked_frames["image"], stacked_frames["mask"] = \
                    self.transform(stacked_frames["image"], stacked_frames["mask"])

                stacked_frames["target_image"], stacked_frames["target_mask"] = \
                    self.transform(stacked_frames["target_image"], stacked_frames["target_mask"])
            
            return stacked_frames


class ObjectDatasetValidation(torch.utils.data.Dataset):
    def __init__(self,
                 path: Union[str, Path],
                 sample_size: int = 32,
                 transform: Optional[Callable] = None):
        self._path = Path(path)
        self.sample_size = sample_size
        self.transform = transform

        self._load_index()

    def _load_index(self) -> None:
        self._samples = []

        for sample_path in self._path.iterdir():
            if sample_path.is_dir():
                self._samples += self._load_sample_frames(sample_path)

    def _load_sample_frames(self, path: Path) -> List[_SampleFrameValidation]:
        frames = []

        for image_path in path.glob("*.image.png"):
            name = image_path.name[:-len(".image.png")]
            frame = _SampleFrameValidation.from_path(image_path.parent, name)

            if frame.is_valid():
                frames.append(frame)

        return frames
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, index: int) -> Dict:
        frame = self._samples[index].load(self.sample_size)

        frame["image"] = frame["image"].unsqueeze(0)
        frame["mask"] = frame["mask"].unsqueeze(0)
        frame["target_image"] = frame["target_image"].unsqueeze(0)
        frame["target_mask"] = frame["target_mask"].unsqueeze(0)
            
        if self.transform is not None:
            frame["image"], frame["mask"] = \
                self.transform(frame["image"], frame["mask"])

            frame["target_image"], frame["target_mask"] = \
                self.transform(frame["target_image"], frame["target_mask"])
        
        return frame
