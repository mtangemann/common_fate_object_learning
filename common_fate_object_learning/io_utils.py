import PIL.Image


def pil_loader(path: str, mode: str = "RGB") -> PIL.Image.Image:
    """open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835).
    
    Reference: https://pytorch.org/vision/0.8/_modules/torchvision/datasets/folder.html#DatasetFolder
    """
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert(mode)
