# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms
import numpy as np
import random
from PIL import Image

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")



class AdaptiveHeadTailCrop:
    def __init__(self, output_size=(98, 98)):  
        self.output_size = output_size  # Final output size
        self.center_crop = transforms.CenterCrop(output_size)  # Center crop transformation

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image, but got {type(img)}")

        # Convert image to grayscale 
        gray = img.convert("L")
        img_array = np.array(gray)

        # Find the first and last non-black pixel
        non_black_pixels = np.where(img_array > 55)  # Threshold to remove pure black
        if len(non_black_pixels[0]) == 0:
            raise ValueError("No non-black pixels found in the image")

        top = np.min(non_black_pixels[0])  # First bright pixel (head)
        bottom = np.max(non_black_pixels[0])  # Last bright pixel (tail)
        
        # Define cropping height dynamically
        grain_height = bottom - top
        crop_h = grain_height // 2  # Crop half of the grain height

        # Crop the head or tail based on detected positions
        head_crop = img.crop((0, top, img.width, top + crop_h))  
        tail_crop = img.crop((0, bottom - crop_h, img.width, bottom))

        # Randomly select head or tail
        selected_crop = random.choice([head_crop, tail_crop])

        # Apply **center crop** to remove excess background
        selected_crop = self.center_crop(selected_crop)

        # # Resize the cropped image to the final output size
        # selected_crop = selected_crop.resize(self.output_size, Image.LANCZOS)

        return selected_crop


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=98,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, ratio=(1, 1), interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # self.geometric_augmentation_local = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             local_crops_size, scale=local_crops_scale, ratio=(1, 1), interpolation=transforms.InterpolationMode.LANCZOS
        #         ),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #     ]
        # )

        self.geometric_augmentation_local = transforms.Compose(
            [
                AdaptiveHeadTailCrop(output_size=(local_crops_size, local_crops_size)),
                transforms.Resize(
                    local_crops_size, interpolation=transforms.InterpolationMode.LANCZOS
                ),  
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )


        intensity_ranges = {
        "brightness": np.linspace(0.9, 1.1, 5),  # Range should be >0
        "contrast": np.linspace(0.9, 1.1, 5),
        "saturation": np.linspace(0.9, 1.1, 5),  # Ensure positive values
        "hue": np.linspace(-0.05, 0.05, 5)  # Hue range should be (-0.5, 0.5)
        }

        # color distorsions 
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=(intensity_ranges["brightness"].min(), intensity_ranges["brightness"].max()),
                                            contrast=(intensity_ranges["contrast"].min(), intensity_ranges["contrast"].max()),
                                            saturation=(intensity_ranges["saturation"].min(), intensity_ranges["saturation"].max()),
                                            hue=(intensity_ranges["hue"].min(), intensity_ranges["hue"].max()))],
                                            p=0.8,),
            ]
        )

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
