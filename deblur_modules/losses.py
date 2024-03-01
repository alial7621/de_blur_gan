from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import MSELoss, Module
import torch
import copy

def hook(module, input, output):
    global intermediate_output
    intermediate_output = output

class PerceptualLoss(Module):
    """
    Perceptual loss is a MSE loss between embedding of two images using VGG16 pre-trained model
    """

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        desired_layer = self.resnet.layer3[-1]
        desired_layer.register_forward_hook(hook)
        self.mse_loss = MSELoss()

    def forward(self, real_image, generated_image):
        self.resnet(real_image)
        real_img_embed = copy.deepcopy(intermediate_output.detach())
        self.resnet(generated_image)
        gen_img_embed = intermediate_output

        return self.mse_loss(real_img_embed, gen_img_embed)


def wasserstein_loss(y_true, disc_output):
    """
    Get critics output for real and generated image and return the loss
    """
    return torch.mean(y_true*disc_output)