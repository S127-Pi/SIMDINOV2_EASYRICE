import torch
import torch.nn as nn

# for 224
def get_dino_finetuned_downloaded(model_path, modelname):
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    # load finetuned weights

    # pos_embed has wrong shape
    if model_path is not None:
        pretrained = torch.load(model_path, map_location=torch.device("cpu"))
        # make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace("backbone.", "")
                new_state_dict[new_key] = value
        # change shape of pos_embed
        input_dims = {
            "dinov2_vits14": 384,
            "dinov2_vits14_reg": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitb14_reg": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitl14_reg": 1024,
            "dinov2_vitg14": 1536,
            "dinov2_vitg14_reg": 153
        }
        pos_embed = nn.Parameter(torch.zeros(1, 257, input_dims[modelname]))
        if model_path == "patch14/eval/training_99999/teacher_checkpoint.pth":
            pos_embed = nn.Parameter(torch.zeros(1, 197, input_dims[modelname]))
            model.pos_embed = pos_embed
        else:
            model.pos_embed = pos_embed
            
        # load state dict
        msg = model.load_state_dict(new_state_dict, strict=True)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(model_path, msg))
    model.to("cuda")
    return model