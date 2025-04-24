from torchvision import models
import torch.nn.functional as F
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
        

class ContentLoss(nn.Module):
    """
    Content loss for image similarity.
    Supports L1, L2, and perceptual losses.
    """
    def __init__(self, loss_type='l1', lambda_content=100.0, perceptual_layers=None):
        """
        Initialize the ContentLoss class.
        
        Args:
            loss_type (str): Type of content loss ('l1', 'l2', 'perceptual')
            lambda_content (float): Weight for content loss
            perceptual_layers (list): VGG layers to use for perceptual loss
        """
        super(ContentLoss, self).__init__()
        self.loss_type = loss_type
        self.lambda_content = lambda_content
        
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        elif loss_type == 'perceptual':
            self.loss = nn.L1Loss()
            self.perceptual_layers = perceptual_layers or ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
            self.vgg = self._get_vgg_model()
        else:
            raise NotImplementedError(f'Content loss type {loss_type} not implemented')
    
    def _get_vgg_model(self):
        """Get pre-trained VGG model for perceptual loss."""
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def _get_vgg_features(self, x, layers):
        """Extract features from VGG model at specified layers."""
        features = {}
        layer_names = {
            'conv1_1': '0', 'conv1_2': '2',
            'conv2_1': '5', 'conv2_2': '7',
            'conv3_1': '10', 'conv3_2': '12', 'conv3_3': '14',
            'conv4_1': '17', 'conv4_2': '19', 'conv4_3': '21',
            'conv5_1': '24', 'conv5_2': '26', 'conv5_3': '28'
        }
        
        # Convert layer names to indices
        indices = [int(layer_names[layer]) for layer in layers if layer in layer_names]
        
        # Extract features
        for i, module in enumerate(self.vgg):
            x = module(x)
            if i in indices:
                layer_name = [k for k, v in layer_names.items() if int(v) == i][0]
                features[layer_name] = x
        
        return features
    
    def forward(self, prediction, target):
        """
        Calculate content loss between prediction and target.
        
        Args:
            prediction: Generated image
            target: Ground truth image
            
        Returns:
            loss: Calculated content loss
        """
        if self.loss_type in ['l1', 'l2']:
            return self.lambda_content * self.loss(prediction, target)
        elif self.loss_type == 'perceptual':
            # Normalize to VGG input range
            prediction = (prediction + 1) / 2  # [-1, 1] -> [0, 1]
            target = (target + 1) / 2
            
            # VGG normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(prediction.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(prediction.device)
            prediction = (prediction - mean) / std
            target = (target - mean) / std
            
            # Get VGG features
            pred_features = self._get_vgg_features(prediction, self.perceptual_layers)
            target_features = self._get_vgg_features(target, self.perceptual_layers)
            
            # Calculate perceptual loss
            perceptual_loss = 0
            for layer in self.perceptual_layers:
                if layer in pred_features and layer in target_features:
                    perceptual_loss += self.loss(pred_features[layer], target_features[layer])
            
            return self.lambda_content * perceptual_loss
        

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