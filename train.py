from deblur_modules.config import args
from deblur_modules.data_loader import GoProDataLoader
from deblur_modules.models import Generator, Discriminator
from deblur_modules.losses import wasserstein_loss, PerceptualLoss

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import numpy as np
import tqdm

# from utils.logger import Logger


# def save_ckpt(path, model, optimizer, scheduler, epoch, best_score):
#     """ save current model
#     """
#     state = {
#         "epoch": epoch,
#         "model_state": model.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#         "scheduler_state": scheduler.state_dict(),
#         "best_score": best_score
#     }
#     torch.save(state, path)


def train():
    
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
    critic = Discriminator()

    # Loss functions and optimizers
    precept_loss = PerceptualLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args["INIT_LR"], 
                                     betas=(args["MOMENTUM1"], args["MOMENTUM2"]), 
                                     weight_decay=args["WEIGHT_DECAY"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args["INIT_LR"], 
                                     betas=(args["MOMENTUM1"], args["MOMENTUM2"]), 
                                     weight_decay=args["WEIGHT_DECAY"])
    

    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    #     checkpoint = torch.load(opts.ckpt, map_location="cpu")
    #     model.load_state_dict(checkpoint["model_state"], strict=True)
    #     optimizer.load_state_dict(checkpoint["optimizer_state"])
    #     scheduler.load_state_dict(checkpoint["scheduler_state"])
    #     cur_epoch = checkpoint["epoch"] + 1
    #     best_score = checkpoint['best_score']
    #     logger.info("[!] Model restored from %s" % opts.ckpt)
    #     # if we want to resume training, resume trainer from checkpoint
    #     if 'trainer_state' in checkpoint:
    #         trainer.load_state_dict(checkpoint['trainer_state'])
    #     del checkpoint
    # else:
    #     if opts.step == 0:
    #         logger.info("[!] Train from scratch")

if __name__ == '__main__':
    train()

