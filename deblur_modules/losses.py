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


class GradientPenalty(nn.Module):
    """
    Gradient Penalty for WGAN-GP.
    Enforces Lipschitz constraint on discriminator.
    """
    def __init__(self, lambda_gp=10.0):
        """
        Initialize the GradientPenalty class.
        
        Args:
            lambda_gp (float): Weight for gradient penalty
        """
        super(GradientPenalty, self).__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator, sharp_images, generated_images):
        """
        Calculate gradient penalty.
        
        Args:
            discriminator: Discriminator model
            sharp_images: Real images
            generated_images: Generated images
            
        Returns:
            gradient_penalty: Calculated gradient penalty
        """
        # Get random interpolation between real and fake samples
        batch_size = sharp_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(sharp_images.device)
        interpolates = alpha * sharp_images + (1 - alpha) * generated_images
        interpolates.requires_grad_(True)
        
        # Calculate discriminator output for interpolated images
        d_interpolates = discriminator(interpolates)
        
        # Get gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty
    

class TVLoss(nn.Module):
    """
    Total Variation Loss for spatial smoothness.
    Encourages spatial smoothness in the generated image.
    """
    def __init__(self, lambda_tv=0.1):
        """
        Initialize the TVLoss class.
        
        Args:
            lambda_tv (float): Weight for TV loss
        """
        super(TVLoss, self).__init__()
        self.lambda_tv = lambda_tv
    
    def forward(self, x):
        """
        Calculate total variation loss.
        
        Args:
            x: Input image
            
        Returns:
            loss: Calculated TV loss
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.lambda_tv * (h_tv + w_tv) / (batch_size * count_h * count_w)
    
    def _tensor_size(self, t):
        """Calculate the total number of elements in a tensor."""
        return t.size()[1] * t.size()[2] * t.size()[3]