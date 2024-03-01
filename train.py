import deblur_modules.config as config
from deblur_modules.data_loader import GoProDataLoader
from deblur_modules.models import Generator, Discriminator
from deblur_modules.losses import wasserstein_loss, PerceptualLoss

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from tqdm import tqdm
import os

# from utils.logger import Logger

def num_params_func(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    num_params = sum([params.numel() for params in model.parameters()])
    num_trainables = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return num_params, num_trainables

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
    if isinstance(args.image_size, int):
        image_size = (args.image_size, args.image_size)
    else:
        image_size = tuple(args.image_size)

    transform = transforms.Compose([
        transforms.Resize(size=image_size)
    ])

    # Initiate models 
    generator = Generator()
    generator.to(device)

    critic = Discriminator()
    critic.to(device)

    num_params, num_trainables = num_params_func(generator)
    print(f"Number of total parameters in the generator model: {num_params}")
    print(f"Number of trainable parameters in the generator model: {num_trainables}\n")

    num_params, num_trainables = num_params_func(critic)
    print(f"Number of total parameters in the discriminator model: {num_params}")
    print(f"Number of trainable parameters in the discriminator model: {num_trainables}\n")

    # Loss functions and optimizers
    percept_loss = PerceptualLoss()
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

        print("Checkpoint loaded")
        print(f"Starting from epoch {cur_epoch}\n")

        # logger.info("[!] Model restored from %s" % args.load_checkpoint)
        del checkpoint
    else:
        pass
        # logger.info("[!] Train from scratch")

    # TODO Save best model and last model. Best model is based on a generator model with the minimum loss

    # Gave as a label to wasserstein loss based on the input of the discriminator to calculate the loss
    true_output, false_output = torch.ones((args.batch_size, 1)).to(device), -torch.ones((args.batch_size, 1)).to(device)
    
    while cur_epoch <= args.epochs:
        print(f"Epoch: {cur_epoch}")
        gen_loss = 0
        critic_loss = 0

        for input_data in tqdm(train_loader):
            input_data[0], input_data[1] = (input_data[0].float()).to(device), (input_data[1].float()).to(device)
            input_data[0] = input_data[0].transpose(1, 3).transpose(2, 3)
            input_data[1] = input_data[1].transpose(1, 3).transpose(2, 3)

            # Resize the images
            sharp_image = transform(input_data[0])
            blury_image = transform(input_data[1])

            # set the generator model grad to zero and get the generated image
            gen_optimizer.zero_grad()
            generated_image = generator(blury_image)
            
            batch_critic_loss = 0
            critic.train()
            for _ in range(args.critic_update):
                
                # train the critic model on original image
                critic_optimizer.zero_grad()
                critic_score_sharp = critic(sharp_image)
                critic_loss_sharp = wasserstein_loss(true_output, critic_score_sharp)
                critic_loss_sharp.backward()
                critic_optimizer.step()

                # train the critic on generated image
                critic_optimizer.zero_grad()
                critic_score_sharp = critic(generated_image)
                critic_loss_gen = wasserstein_loss(false_output, critic_score_sharp)
                critic_loss_gen.backward()
                critic_optimizer.step()

                overall_critic_loss = 0.5 * (critic_loss_gen + critic_loss_sharp)
                batch_critic_loss += overall_critic_loss.item()

            # Overall critic loss for this batch
            critic_loss += (batch_critic_loss / args.critic_update)

            # get the critic score and loss
            critic.eval()
            with torch.no_grad():
                critic_score = critic(generated_image)
            critic_loss_gen = wasserstein_loss(true_output, critic_score)

            # get the perceptual loss and overall loss for generator
            perceptual_loss = percept_loss(sharp_image, generated_image)
            overall_generator_loss = (10 * perceptual_loss) + critic_loss_gen
            gen_loss += overall_generator_loss.item()

            # train the generator model
            overall_generator_loss.backward()
            gen_optimizer.step()

        cur_epoch += 1
        gen_loss = gen_loss / len(train_loader)
        critic_loss = critic_loss / len(train_loader)

        gen_loss_history.append(gen_loss)
        critic_loss_history.append(critic_loss)

        gen_scheduler.step(gen_loss)
        critic_scheduler.step(critic_loss)
        

        
if __name__ == '__main__':

    parser = config.get_argparser()

    args = parser.parse_args()

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
        os.mkdir(args.checkpoint_dir + "/models")

    train(args)

