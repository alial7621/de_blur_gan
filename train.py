import deblur_modules.config as config
from deblur_modules.data_loader import GoProDataLoader
from deblur_modules.models import Generator, Discriminator
from deblur_modules.losses import wasserstein_loss, PerceptualLoss

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import numpy as np
import tqdm
import os

# from utils.logger import Logger


def save_ckpt(path, gen_model, critic_model, gen_optim, critic_optim, 
              gen_schdlr, critic_schdlr, epoch, gen_loss_history, critic_loss_history):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "gen_state": gen_model.state_dict(),
        "critic_state": critic_model.state_dict(),
        "gen_optim_state": gen_optim.state_dict(),
        "critic_optim_state": critic_optim.state_dict(),
        "gem_schdlr_state": gen_schdlr.state_dict(),
        "critic_schdlr_state": critic_schdlr.state_dict(),
        "gen_loss_history": gen_loss_history,
        "critic_loss_history": critic_loss_history
    }
    torch.save(state, path)


def train(args):
    
    if args.use_manual_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare datalaoders
    train_data = GoProDataLoader(root_path=args.data_dir, dataset_path=args.dataset_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Resize augmentation
    if type(args.image_size) == int:
        image_size = (args.image_size, args.image_size)
    else:
        image_size = tuple(args.image_size)
    transform = v2.Compose([
        v2.Resize(size=image_size)
    ])

    # Initiate models 
    generator = Generator()
    generator.to(device)

    critic = Discriminator()
    critic.to(device)       

    # Loss functions and optimizers
    precept_loss = PerceptualLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.init_lr, 
                                        betas=(args.momentum1, args.momentum2))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.init_lr, 
                                        betas=(args.momentum1, args.momentum2))

    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, mode="min", factor=args.weight_decay_factor,
                                                patience=args.lr_wait, threshold=args.lr_thresh,
                                                threshold_mode="abs", min_lr=args.final_lr, verbose=True)  
    critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode="min", factor=args.weight_decay_factor,
                                                patience=args.lr_wait, threshold=args.lr_thresh,
                                                threshold_mode="abs", min_lr=args.final_lr, verbose=True)  

    cur_epoch = 1
    gen_loss_history = []
    critic_loss_history = []

    if args.load_checkpoint is not None and os.path.isfile(args.load_checkpoint):
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu")
        generator.load_state_dict(checkpoint["gen_state"], strict=True)
        critic.load_state_dict(checkpoint["critic_state"], strict=True)

        gen_optimizer.load_state_dict(checkpoint["gen_optim_state"])
        critic_optimizer.load_state_dict(checkpoint["critic_optim_state"])
        
        gen_scheduler.load_state_dict(checkpoint["gem_schdlr_state"])
        critic_scheduler.load_state_dict(checkpoint["critic_schdlr_state"])

        cur_epoch = checkpoint["epoch"] + 1
        gen_loss_history = checkpoint['gen_loss_history']
        critic_loss_history = checkpoint['critic_loss_history']

        # logger.info("[!] Model restored from %s" % args.load_checkpoint)
        del checkpoint
    else:
        pass
        # logger.info("[!] Train from scratch")

    # TODO Save best model and last model. Best model is based on a generator model with the minimum loss
        
if __name__ == '__main__':

    parser = config.get_argparser()

    args = parser.parse_args()

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
        os.mkdir(args.checkpoint_dir + "/models")

    train(args)

