import timm
import torch
import segmentation_models_pytorch as smp

from torch import nn

MODELS = {
    'Unet': smp.Unet,
    'Unet++': smp.UnetPlusPlus,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet,
    'DeepLabV3': smp.DeepLabV3,
    'DeepLabV3+': smp.DeepLabV3Plus,
    'Linknet': smp.Linknet,
    'MAnet': smp.MAnet,
    'PAN': smp.PAN
}

class TorchModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model = MODELS[model_config['smp_model_name']](
            **model_config['kwargs']
        )

    def forward(self, x):
        mask = self.model(x)
        label = torch.amax(mask, dim=(1,2,3))
        return mask, label
