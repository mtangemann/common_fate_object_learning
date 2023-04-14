# https://pytorch.org/vision/0.8/_modules/torchvision/utils.html#make_grid
def normalize(tensor, value_range = None, scale_each = False):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if value_range is not None:
        assert isinstance(value_range, tuple), \
            "value_range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)
    
    return tensor
