# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import sys
import os
from .masking import MaskingGenerator
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')))
# from simdinov2.data import MaskingGenerator


def is_patch_black(patch, threshold=0.6):
    """
    Check if a patch is entirely black in the normalized ImageNet range.
    
    A patch is considered black if all pixels are very close to the expected normalized black value.
    """


    # ImageNet normalization constants
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=patch.device).view(3, 1, 1)  # (C, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=patch.device).view(3, 1, 1)  # (C, 1, 1)
    
    # Denormalize the patch
    denormalized_patch = patch * imagenet_std + imagenet_mean
    
    # Check if all values are very close to the black value (0, 0, 0)
    black_value = torch.tensor([0.0, 0.0, 0.0], device=patch.device).view(3, 1, 1)
    return torch.all(torch.abs(denormalized_patch - black_value) < threshold)


def mask_local_crops(crops, mask_ratio_tuple, mask_probability,n_tokens,patch_size=14):
     
    B = len(crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []

    _, _, H, W = crops.shape  # Batch size, Channels, Height, Width
    H_patches = H // patch_size  # Number of patches along height
    W_patches = W // patch_size  # Number of patches along width
    Total_patches = H_patches * W_patches  # Total patches per image


    for i in range(B):
            non_black_patches = []
            patch_idx = 0
            pixel_indices = [] 

            # Extract patches and check if they are black
            for row in range(H_patches):
                for col in range(W_patches):
                    r_start = row * patch_size
                    c_start = col * patch_size

                    # Extract patch (C, patch_size, patch_size)
                    patch = crops[i, :, r_start:r_start + patch_size, c_start:c_start + patch_size]
                    
                    if not is_patch_black(patch):  # If patch is not black, add to list
                        non_black_patches.append(patch_idx)

                    patch_idx += 1  # Track patch index
                        # Mask only non-black patches
                        
            if non_black_patches:
                non_black_patches = non_black_patches[2:-1]
                num_mask_patches = int(len(non_black_patches) * random.uniform(*mask_ratio_tuple))
                masked_indices = random.sample(non_black_patches, num_mask_patches)

                mask = torch.zeros(Total_patches, dtype=torch.bool)  # Default all patches to be unmasked (0)
                mask[masked_indices] = 1  # Set chosen patches to be masked (1)
            else:
                mask = torch.zeros(Total_patches, dtype=torch.bool)  # No masking if all patches are black
            
            mask = mask.view(H_patches, W_patches)  # Reshape to original image divided by patch size
            masks_list.append(mask)  # Append mask to list
    random.shuffle(masks_list)
    masks_list = random.choices(masks_list, k=n_samples_masked)

    mask_generator = MaskingGenerator(
    input_size=(H // patch_size, W // patch_size),
    max_num_patches=0.1 * H // patch_size * H // patch_size,
    ) # mask generator for local crops
    for i in range(0, n_samples_masked):
        masks_list.append(torch.BoolTensor(mask_generator(num_masking_patches=0))) # no mask

    collated_masks = torch.stack(masks_list).flatten(1) 

    mask_indices_list = collated_masks.flatten().nonzero().flatten() # get all masked indices
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return collated_masks, mask_indices_list, masks_weight

# def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None, drop_masks=False):
#     # dtype = torch.half  # TODO: Remove

#     n_global_crops = len(samples_list[0][0]["global_crops"])
#     n_local_crops = len(samples_list[0][0]["local_crops"])

#     collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

#     collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

#     B = len(collated_global_crops)
#     N = n_tokens
#     n_samples_masked = int(B * mask_probability)
#     probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
#     upperbound = 0
#     masks_list = []
#     if drop_masks:
#         len_keep = int(N * (1 - mask_probability))
        
#         noise = torch.rand(B, N)  # noise in [0, 1]
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#         # keep the first subset
#         #ids_keep = ids_shuffle[:, :len_keep]
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([B, N], dtype=torch.bool)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         collated_masks = torch.gather(mask, dim=1, index=ids_restore)
#         upperbound = N*B*mask_probability
#     else:
#         for i in range(0, n_samples_masked):
#             prob_min = probs[i]
#             prob_max = probs[i + 1]
#             masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
#             upperbound += int(N * prob_max)
#         for i in range(n_samples_masked, B):
#             masks_list.append(torch.BoolTensor(mask_generator(0)))
#         random.shuffle(masks_list)
#         collated_masks = torch.stack(masks_list).flatten(1)
#     mask_indices_list = collated_masks.flatten().nonzero().flatten()

#     masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
 
#     return {
#         "collated_global_crops": collated_global_crops.to(dtype),
#         "collated_local_crops": collated_local_crops.to(dtype),
#         "collated_masks": collated_masks,
#         "mask_indices_list": mask_indices_list,
#         "masks_weight": masks_weight,
#         "upperbound": upperbound,
#         "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
#     }






def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None, drop_masks=False, patch_size=14):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []


    _, _, H, W = collated_global_crops.shape  # Batch size, Channels, Height, Width
    H_patches = H // patch_size  # Number of patches along height
    W_patches = W // patch_size  # Number of patches along width
    Total_patches = H_patches * W_patches  # Total patches per image

    if drop_masks:
        len_keep = int(N * (1 - mask_probability))
        
        noise = torch.rand(B, N)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        #ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], dtype=torch.bool)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        collated_masks = torch.gather(mask, dim=1, index=ids_restore)
        upperbound = N*B*mask_probability
    else:

        for i in range(B):
                    non_black_patches = []
                    patch_idx = 0

                    # Extract patches and check if they are black
                    for row in range(H_patches):
                        for col in range(W_patches):
                            r_start = row * patch_size
                            c_start = col * patch_size

                            # Extract patch (C, patch_size, patch_size)
                            patch = collated_global_crops[i, :, r_start:r_start + patch_size, c_start:c_start + patch_size]
                            
                            if not is_patch_black(patch):  # If patch is not black, add to list
                                non_black_patches.append(patch_idx)
                            patch_idx += 1  # Track patch index
                               # Mask only non-black patches
                               
                    if non_black_patches:
                        non_black_patches = non_black_patches[5:-5]
                        num_mask_patches = int(len(non_black_patches) * random.uniform(*mask_ratio_tuple))
                        masked_indices = random.sample(non_black_patches, num_mask_patches)

                        mask = torch.zeros(Total_patches, dtype=torch.bool)  # Default all patches to be unmasked (0)
                        mask[masked_indices] = 1  # Set chosen patches to be masked (1)
                    else:
                        mask = torch.zeros(Total_patches, dtype=torch.bool)  # No masking if all patches are black
                    
                    mask = mask.view(H_patches, W_patches)  # Reshape to original image divided by patch size
                    masks_list.append(mask)  # Append mask to list
        masks_list = random.choices(masks_list, k=n_samples_masked)

        # for i in range(0, n_samples_masked):
        #     prob_min = probs[i]
        #     prob_max = probs[i + 1]
        #     masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        #     upperbound += int(N * prob_max)
        for i in range(0, n_samples_masked):
            masks_list.append(torch.BoolTensor(mask_generator(0)))
        random.shuffle(masks_list)
        collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    collated_local_mask, local_mask_indices_list, local_masks_weight = mask_local_crops(collated_local_crops, mask_ratio_tuple, mask_probability, n_tokens, patch_size=14)
    
    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
