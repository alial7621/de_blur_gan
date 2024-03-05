from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torch.nn import MSELoss, Module
import torch

class PerceptualLoss(Module):
    """
    Perceptual loss is an MSE loss between the intermediate embeddings of two images using a pre-trained VGG16 model.
    """

    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).features.to(device)
        self.mse_loss = MSELoss()

    def forward(self, sharp_images, generated_images):
        real_img_embed = self.vgg16(sharp_images)
        gen_img_embed = self.vgg16(generated_images)

        return self.mse_loss(real_img_embed, gen_img_embed)

def gradient_penalty(gradient):
    """
    This function calculates the gradient penalty term, which is used to enforce the Lipschitz constraint in Wasserstein GANs.
    The source of the function: https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan
    """
    gradient = gradient.reshape(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)
    
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty

def get_gradient(critic, sharp_images, generated_images, epsilon):
    """
    This function computes the gradient of the critic's scores with respect to the mixed images.
    The source of the function: https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan
    """

    mixed_images = sharp_images * epsilon + generated_images * (1 - epsilon)

    mixed_scores = critic(mixed_images)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True 
    )[0]
    return gradient

def wasserstein_loss(critic_score_gen, critic_score_sharp, gp, c_lambda):
    """
    This function computes the critic loss for training the critic in a Wasserstein GAN.
    The source of the function: https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan
    """
    critic_loss = torch.mean(critic_score_gen) - torch.mean(critic_score_sharp) + c_lambda * gp
    return critic_loss

def wasserstein_loss_generator(critic_score_gen):
    """
    This function computes the critic loss for training the generator in a Wasserstein GAN.
    The source of the function: https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan
    """
    gen_loss = -1. * torch.mean(critic_score_gen)
    return gen_loss