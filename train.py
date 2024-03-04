import deblur_modules.config as config
from deblur_modules.data_loader import GoProDataLoader
from deblur_modules.models import Generator, Discriminator
from deblur_modules.losses import get_gradient, gradient_penalty, wasserstein_loss, \
                                  wasserstein_loss_generator, PerceptualLoss

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def num_params_func(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    num_params = sum([params.numel() for params in model.parameters()])
    num_trainables = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return num_params, num_trainables

def save_ckpt(path, gen_model, critic_model, gen_optim, critic_optim, 
              gen_schdlr, critic_schdlr, epoch, gen_loss_history, critic_loss_history):
    """ 
    save a checkpoint of the models
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
        transforms.Resize(size=image_size, antialias=True)
    ])

    # Initiate models 
    generator = Generator()
    critic = Discriminator()
    # set the device for both models
    generator.to(device)
    critic.to(device)

    num_params, num_trainables = num_params_func(generator)
    print(f"Number of total parameters in the generator model: {num_params}")
    print(f"Number of trainable parameters in the generator model: {num_trainables}\n")

    num_params, num_trainables = num_params_func(critic)
    print(f"Number of total parameters in the discriminator model: {num_params}")
    print(f"Number of trainable parameters in the discriminator model: {num_trainables}\n")

    # Loss functions and optimizers
    percept_loss = PerceptualLoss(device=device)
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
    best_loss = float('inf')

    if args.load_checkpoint is not None and os.path.isfile(args.load_checkpoint):
        print("A checkpoint detected")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint["gen_state"], strict=True)
        critic.load_state_dict(checkpoint["critic_state"], strict=True)

        gen_optimizer.load_state_dict(checkpoint["gen_optim_state"])
        critic_optimizer.load_state_dict(checkpoint["critic_optim_state"])
        
        gen_scheduler.load_state_dict(checkpoint["gem_schdlr_state"])
        critic_scheduler.load_state_dict(checkpoint["critic_schdlr_state"])

        cur_epoch = checkpoint["epoch"] + 1
        gen_loss_history = checkpoint['gen_loss_history']
        critic_loss_history = checkpoint['critic_loss_history']
        best_loss = min(gen_loss_history)

        print("The checkpoint loaded")
        print(f"Starting from epoch {cur_epoch}\n")

        del checkpoint
    else:
        print("There is no checkpoint. The models will be trained from the scratch")
    
    while cur_epoch <= args.epochs:
        print(f"Epoch: {cur_epoch}")
        gen_loss = 0
        critic_loss = 0
        generator.train()
        
        for input_data in tqdm(train_loader):
            input_data[0], input_data[1] = (input_data[0].float()).to(device), (input_data[1].float()).to(device)
            # Resize the images
            sharp_image = transform(input_data[0])
            blury_image = transform(input_data[1])

            # set the generator model grad to zero and get the generated image
            gen_optimizer.zero_grad()
            generated_image = generator(blury_image)
            
            # The source of the following code for training the critic model is https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan 
            critic.train()
            for _ in range(args.critic_update):  
                # train the critic model on the original image
                critic_optimizer.zero_grad()
                critic_score_sharp = critic(sharp_image)
                critic_score_gen = critic(generated_image)

                epsilon = torch.rand(len(sharp_image), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(critic, sharp_image, generated_image, epsilon)
                gp = gradient_penalty(gradient)
                final_critic_loss = wasserstein_loss(critic_score_gen, critic_score_sharp, gp, args.c_lambda)

                final_critic_loss.backward(retain_graph=True)
                critic_optimizer.step()

                critic_loss += (final_critic_loss.item() / args.critic_update)

            # get the critic score and loss
            critic.eval()
            with torch.no_grad():
                critic_score = critic(generated_image)
            critic_loss_gen = wasserstein_loss_generator(critic_score)

            # get the perceptual loss and overall loss for generator
            perceptual_loss = percept_loss(sharp_image, generated_image)
            overall_generator_loss = (args.percept_weight * perceptual_loss) + critic_loss_gen
            gen_loss += overall_generator_loss.item()

            # train the generator model
            overall_generator_loss.backward()
            gen_optimizer.step()
            
        gen_loss = gen_loss / len(train_loader)
        critic_loss = critic_loss / len(train_loader)
        print(f"Loss for the generator model: {gen_loss}, Loss for the critic model: {critic_loss}")
        
        gen_loss_history.append(gen_loss)
        critic_loss_history.append(critic_loss)

        gen_scheduler.step(gen_loss)
        critic_scheduler.step(critic_loss)
        
        # save the best and last model
        generator.eval()
        save_ckpt(args.checkpoint_dir + "/models/last_model.pt", gen_model=generator, critic_model=critic,
                  gen_optim=gen_optimizer, critic_optim=critic_optimizer, gen_schdlr=gen_scheduler, critic_schdlr=critic_scheduler,
                  epoch=cur_epoch, gen_loss_history=gen_loss_history, critic_loss_history=critic_loss_history)
        if gen_loss < best_loss:
            best_loss = gen_loss
            # For the best model, we just need to save the generator model
            torch.save(generator.state_dict(), args.checkpoint_dir + "/models/best_model.pt")

        cur_epoch += 1

    # Draw and save the loss history diagram
    plt.figure()
    plt.title("Loss Curve for the Generator and the Critic Models")
    plt.xlabel("Step No.")
    plt.ylabel("Loss value")
    plt.plot(list(range(1, len(gen_loss_history)+1)), gen_loss_history, "blue", label="Generator")
    plt.plot(list(range(1, len(critic_loss_history)+1)), critic_loss_history, "red", label="Critic")
    plt.legend()
    plt.savefig(args.checkpoint_dir + "/loss_history/Loss history.png")
    plt.close()
        
if __name__ == '__main__':

    parser = config.get_argparser()

    args = parser.parse_args()

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
        os.mkdir(args.checkpoint_dir + "/models")
        os.mkdir(args.checkpoint_dir + "/loss_history")

    train(args)

