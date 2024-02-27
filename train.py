from deblur_modules.config import args
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


def train(opts):
    
    if args['USE_MAN_SEED']:
        np.random.seed(args["SEED"])
        torch.manual_seed(args["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare datalaoders
    train_data = GoProDataLoader(root_path=args["DATA_DIR"], dataset_path=args["DATASET_DIR"])
    train_loader = DataLoader(train_data, batch_size=args["BATCH_SIZE"], shuffle=True)

    # Resize augmentation
    transform = v2.Compose([
        v2.Resize(size=args["IMAGE_SIZE"])
    ])

    # Initiate models 
    generator = Generator()
    generator.to(device)

    critic = Discriminator()
    critic.to(device)

    # Loss functions and optimizers
    precept_loss = PerceptualLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args["INIT_LR"], 
                                     betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args["INIT_LR"], 
                                     betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
  
    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, mode="min", factor=args["WEIGHT_DECAY"],
                                             patience=args["LR_WAIT"], threshold=args["LR_THRESH"],
                                             threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)  
    critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode="min", factor=args["WEIGHT_DECAY"],
                                             patience=args["LR_WAIT"], threshold=args["LR_THRESH"],
                                             threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)  
    

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        if opts.step == 0:
            logger.info("[!] Train from scratch")

if __name__ == '__main__':
    
    parser = config.get_argparser()

    opts = parser.parse_args()
    opts = config.modify_command_options(opts)

    # Create checkpoint directory
    if not os.path.exists(args["CODE_DIR"] + "/checkpoints"):
        os.mkdir(args["CODE_DIR"] + "/checkpoints")
        os.mkdir(args["CODE_DIR"] + "/checkpoints/models")

    train(opts)

