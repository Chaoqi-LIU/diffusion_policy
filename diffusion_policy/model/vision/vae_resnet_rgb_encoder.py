from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import einops
import torchvision
import torch.nn.functional as F
from diffusers import AutoencoderKLTemporalDecoder
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_nets as rmon
from diffusion_policy.common.pytorch_util import replace_submodules


from typing import Dict, Tuple, Union

class RobomimicRgbEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # init global state
        ObsUtils.initialize_obs_modality_mapping_from_dict({"rgb": "agentview_image"})

        obs_encoder = rmon.ObservationEncoder()
        net = rmbn.VisualCore(
            input_shape=(4, 32, 32),
            feature_dimension=64,
            backbone_class='ResNet18Conv',
            backbone_kwargs={
                'input_channels': 4,
                'input_coord_conv': False,
            },
            pool_class='SpatialSoftmax',
            pool_kwargs={
                'num_kp': 32,
                'temperature': 1.0,
                'noise': 0.0,
            },
            flatten=True,
        )
        obs_encoder.register_obs_key(
            name="agentview_image",
            shape=(4, 32, 32),
            net=net,
        )

        # use group norm instead of batch norm
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features//16,
                num_channels=x.num_features,
            )
        )

        obs_encoder.make()
        self.encoder = obs_encoder

    def forward(self, obs_dict):
        return self.encoder(obs_dict)
    
    
    @torch.no_grad()
    def output_feature_dim(self):
        return self.encoder.output_shape()[0]



SVD_SCALE = 0.18215
class VaeResNetRgbEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae")
        self.vae = vae.to(device='cuda', dtype=torch.bfloat16).eval()
        self.resnet = RobomimicRgbEncoder().to("cuda")

        for param in self.vae.parameters():
            param.requires_grad = False
        
    def forward(self, obs_dict):
        image = obs_dict['agentview_image']
        assert image.shape[1:] == (3, 84, 84), image.shape

        # normalize image
        image = image / 255.0
        image = F.interpolate(image, size=(256, 256), mode='bilinear')
        assert image.shape[1:] == (3, 256, 256), image.shape
        image = (image * 2.) - 1.

        # encode image with vae
        latent = self.vae.encode(image.to(dtype=torch.bfloat16)).latent_dist.mean * SVD_SCALE
        latent = latent.to(dtype=torch.float32)
        assert latent.shape[1:] == (4, 32, 32), latent.shape

        # encode latent with resnet
        latent = self.resnet({"agentview_image": latent})
        return latent
    
    @torch.no_grad()
    def output_feature_dim(self):
        return self.resnet.output_feature_dim()