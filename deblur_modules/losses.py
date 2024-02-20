from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torch

class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss is a MSE loss between embedding of two images using VGG16 pre-trained model
    """

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.VGG16 = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, real_image, generated_image):
        real_img_embed = self.VGG16(real_image)
        gen_img_embed = self.VGG16(generated_image)

        return self.mse_loss(real_img_embed, gen_img_embed)


def wasserstein_loss(disc_output_real, disc_output_gen):
    """
    Get critics output for real and generated image and return the loss
    """
    return torch.mean(disc_output_real*disc_output_gen)