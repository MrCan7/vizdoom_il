import random
import torch
from torchvision import transforms
from torchvision.transforms import Lambda, Compose
from torchvision.transforms import functional as FF

"""
    Older Implementation of torch color jitter
    Needed for fixing paramters per sequence.
    Otherwise per image params are applied.
"""
class StaticColorJitter(transforms.ColorJitter):
    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []


        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: FF.adjust_brightness(img, brightness_factor)))


        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: FF.adjust_contrast(img, contrast_factor)))


        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: FF.adjust_saturation(img, saturation_factor)))


        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: FF.adjust_hue(img, hue_factor)))


        random.shuffle(transforms)
        transform = Compose(transforms)


        return transform
