# from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import MSELoss, Module
import torch
from torch import nn

class GANLoss(nn.Module):
    """
    GAN loss base class.
    Supports different GAN loss formulations.
    """
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        """
        Initialize the GANLoss class.
        
        Args:
            gan_mode (str): Type of GAN loss ('vanilla', 'lsgan', 'wgangp')
            target_real_label (float): Value for real label
            target_fake_label (float): Value for fake label
        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgangp':
            self.loss = None  # No loss function, handled in forward
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction, target_is_real):
        """
        Calculate loss given discriminator predictions and if target is real/fake.
        
        Args:
            prediction: Discriminator predictions
            target_is_real: Whether the target is real or fake
            
        Returns:
            loss: Calculated loss
        """
        if self.gan_mode == 'wgangp':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        
        
class PerceptualLoss(Module):
    """
    Perceptual loss is an MSE loss between the intermediate embeddings of two images using a pre-trained ResNet model.
    """

    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.resnet50 = resnet50(weights="IMAGENET1K_V1").to(device)
        self.resnet50.eval()
        self.res_feat = nn.Sequential(*list(self.resnet50.children())[:-2])
        # self.vgg16 = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).features.to(device)
        self.mse_loss = MSELoss()

    def forward(self, sharp_images, generated_images, val=False):
        if val:
            with torch.no_grad():
                real_img_embed = self.res_feat(sharp_images)
                gen_img_embed = self.res_feat(generated_images)

        else:
            real_img_embed = self.res_feat(sharp_images)
            gen_img_embed = self.res_feat(generated_images)

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