# Unsupervised Object Learning via Common Fate

This repository contains the code for training the model proposed in the CLeaR 2023
paper [Unsupervised Object Learning via Common Fate](https://arxiv.org/abs/2110.06562).

The code for creating the Fishbowl dataset proposed in the same paper is available at
https://github.com/amazon-science/common-fate-fishbowl.


## Prerequesites
Install all dependencies listed in `requirements.txt`.


## Usage

### Object Model
Before training the object model, crops of the candidate objects have to be extracted
based on the ground truth occluded masks or the motion segmentation:

```bash
# ground truth occluded masks
python extract_object_crops.py \
  --dataset-path data/fishbowl-train \
  --segmentation-path data/fishbowl-train \
  --segmentation-type json \
  --output-path data/fishbowl-train-objects

# motion segmentation
python extract_object_crops.py \
  --dataset-path data/fishbowl-train \
  --segmentation-path output/moseg/fishbowl-train \
  --segmentation-type png \
  --output-path output/moseg/fishbowl-train-objects
```

Adapt the paths to the extracted objects in the object model config file if necesseray,
then train the model using:

```bash
python object_model.py train \
    --config configs/object_model.yaml \
    --output-path output/object_model
```

Objects can be sampled from the model using:

```bash
python object_model.py sample \
    --config configs/object_model.yaml \
    --checkpoint-path output/object_model/checkpoint_last.pth \
    --output-path output/object_model_sample.png
```


## Copyright note
The VAE training code is based on https://github.com/AntixK/PyTorch-VAE. The original
[README](common_fate_object_learning/ORIGINAL_README.md) and
[LICENSE](common_fate_object_learning/ORIGINAL_LICENSE.md) are included in this repository.

This work is licensed under the [MIT License](LICENSE).
