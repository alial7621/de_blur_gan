import deblur_modules.config as config
from deblur_modules.data_loader import get_data_loaders
from deblur_modules.models import Generator, Discriminator
from deblur_modules.metrics import PSNR, SSIM
from deblur_modules.losses import get_gradient, gradient_penalty, wasserstein_loss, \
                                  wasserstein_loss_generator, PerceptualLoss

import torch

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
              gen_schdlr, critic_schdlr, epoch, gen_train_loss_history, critic_loss_history,
              test_loss_history, train_SSIM, train_PSNR, test_SSIM, test_PSNR):
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
        "gen_train_loss_history": gen_train_loss_history,
        "critic_loss_history": critic_loss_history,
        "test_loss_history": test_loss_history,
        "train_SSIM_history": train_SSIM,
        "test_SSIM_history": test_SSIM,
        "train_PSNR_history": train_PSNR,
        "test_PSNR_history": test_PSNR
    }
    torch.save(state, path)


def train(args):
    
    if args.use_manual_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare datalaoders
    train_loader, test_loader = get_data_loaders(root_path=args.data_dir dataset_path=args.dataset_dir,
                                                 batch_size=args.batch_size, image_size=args.image_size)

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
    percept_loss = PerceptualLoss(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.init_lr, 
                                        betas=(args.momentum1, args.momentum2))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.init_lr*0.01, 
                                        betas=(args.momentum1, args.momentum2))

    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, mode="max", factor=args.weight_decay_factor,
                                                patience=args.lr_wait, threshold=args.lr_thresh,
                                                threshold_mode="abs", min_lr=args.final_lr, verbose=True)  
    critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode="min", factor=args.weight_decay_factor,
                                                patience=args.lr_wait, threshold=args.lr_thresh,
                                                threshold_mode="abs", min_lr=args.final_lr, verbose=True)  

    cur_epoch = 1
    gen_train_loss_history = []
    gen_test_loss_history = []
    critic_loss_history = []

    train_SSIM_history = []
    test_SSIM_history = []
    train_PSNR_history = []
    test_PSNR_history = []
    best_SSIM = float('-inf')

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
        gen_train_loss_history = checkpoint['gen_train_loss_history']
        critic_loss_history = checkpoint['critic_loss_history']
        gen_test_loss_history = checkpoint['test_loss_history']

        train_SSIM_history = checkpoint['train_SSIM_history']
        test_SSIM_history = checkpoint['test_SSIM_history']
        train_PSNR_history = checkpoint['train_PSNR_history']
        test_PSNR_history = checkpoint['test_PSNR_history']

        best_SSIM = max(test_SSIM_history)

        print("The checkpoint loaded")
        print(f"Starting from epoch {cur_epoch}\n")

        del checkpoint
    else:
        print("There is no checkpoint. The models will be trained from the scratch")

    while cur_epoch <= args.epochs:
        print(f"Epoch: {cur_epoch} of {args.epochs}")
        gen_loss = 0
        critic_loss = 0
        gen_test_loss = 0

        train_SSIM = 0
        train_PSNR = 0
        test_SSIM = 0
        test_PSNR = 0

        generator.train()
        critic.train()
        
        for input_data in tqdm(train_loader):
            sharp_images, blury_images = (input_data[0].float()).to(device), (input_data[1].float()).to(device)
            
            ###########################
            # Train Critic
            ###########################
            critic_optimizer.zero_grad()
            generated_images = generator(blury_images).detach()

            # The source of the following code for training the critic model is https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan  
            for _ in range(args.critic_update):  
                # train the critic model on the original image
                critic_score_sharp = critic(sharp_images)
                critic_score_gen = critic(generated_images)

                epsilon = torch.rand(len(sharp_images), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(critic, sharp_images, generated_images, epsilon)
                gp = gradient_penalty(gradient)
                final_critic_loss = wasserstein_loss(critic_score_gen, critic_score_sharp, gp, args.c_lambda)

                final_critic_loss.backward()

                critic_optimizer.step()

                critic_loss += (final_critic_loss.item() / args.critic_update)

            # Compute metrics
            train_SSIM += SSIM(sharp_images, generated_images, device)
            train_PSNR += PSNR(sharp_images, generated_images, device)

            ###########################
            # Train Generator
            ###########################
            gen_optimizer.zero_grad()

            # Forward pass through generator
            generated_images = generator(blury_images)

            # get the critic score and loss
            critic_score = critic(generated_images).detach()
            critic_loss_gen = wasserstein_loss_generator(critic_score)

            # get the perceptual loss and overall loss for generator
            perceptual_loss = percept_loss(sharp_images, generated_images)
            overall_generator_loss = (args.percept_weight * perceptual_loss) + critic_loss_gen
            gen_loss += overall_generator_loss.item()

            # train the generator model
            overall_generator_loss.backward()
            gen_optimizer.step()
            
        # Final training loss 
        gen_loss = gen_loss / len(train_loader)
        critic_loss = critic_loss / len(train_loader)
        gen_train_loss_history.append(gen_loss)
        critic_loss_history.append(critic_loss)

        # Training Evaluation
        train_SSIM = train_SSIM / len(train_loader)
        train_PSNR = train_PSNR / len(train_loader)
        train_SSIM_history.append(train_SSIM)
        train_PSNR_history.append(train_PSNR)

        print("Step/Epoch: %03d || Generator Train Loss: %.2f  Critic Loss: %.2f || SSIM Train: %.4f  PSNR Train: %.4f"
              %(cur_epoch, gen_loss, critic_loss, train_SSIM, train_PSNR))

        # Evaluate the model on Test/Val data
        generator.eval()

        for input_data in tqdm(test_loader):
            sharp_images, blury_images = (input_data[0].float()).to(device), (input_data[1].float()).to(device)

            with torch.no_grad():
                generated_images = generator(blury_images)
            
            test_SSIM += SSIM(sharp_images, generated_images, device)
            test_PSNR += PSNR(sharp_images, generated_images, device)

            gen_test_loss += percept_loss(sharp_images, generated_images, val=True)
            

        test_SSIM = test_SSIM / len(test_loader)
        test_PSNR = test_PSNR / len(test_loader)
        gen_test_loss = gen_test_loss / len(test_loader)

        # Add to history
        test_SSIM_history.append(test_SSIM)
        test_PSNR_history.append(test_PSNR)
        gen_test_loss_history.append(gen_test_loss)

        print("Generator Test Loss: %.2f|| SSIM Test: %.4f  PSNR Test: %.4f\n"
              %(gen_test_loss, test_SSIM, test_PSNR))

        gen_scheduler.step(test_SSIM)
        critic_scheduler.step(critic_loss)
        
        # save the best and last model
        save_ckpt(args.checkpoint_dir + "/models/last_model.pt", gen_model=generator, critic_model=critic,
                  gen_optim=gen_optimizer, critic_optim=critic_optimizer, gen_schdlr=gen_scheduler, critic_schdlr=critic_scheduler,
                  epoch=cur_epoch, gen_train_loss_history=gen_train_loss_history, critic_loss_history=critic_loss_history,
                  test_loss_history=gen_test_loss_history, train_SSIM=train_SSIM_history, test_SSIM=test_SSIM_history,
                  train_PSNR=train_PSNR_history, test_PSNR=test_PSNR_history)
        if test_SSIM > best_SSIM:
            best_SSIM = test_SSIM
            # For the best model, we just need to save the generator model
            torch.save(generator.state_dict(), args.checkpoint_dir + "/models/best_model.pt")

        cur_epoch += 1


    # Draw and save the loss history diagram
    plt.figure()
    plt.title("Train Loss Curve for the Generator and the Critic Models")
    plt.xlabel("Step No.")
    plt.ylabel("Loss value")
    plt.plot(list(range(1, len(gen_train_loss_history)+1)), gen_train_loss_history, "blue", label="Generator")
    plt.plot(list(range(1, len(critic_loss_history)+1)), critic_loss_history, "red", label="Critic")
    plt.legend()
    plt.savefig(args.checkpoint_dir + "/history/Loss history.png")
    plt.close()
    
    plt.figure()
    plt.title("SSIM and PSNR Curve for Training Data")
    plt.xlabel("Step No.")
    plt.ylabel("Value")
    plt.plot(list(range(1, len(train_SSIM_history)+1)), train_SSIM_history, "blue", label="SSIM")
    plt.plot(list(range(1, len(train_PSNR_history)+1)), train_PSNR_history, "red", label="PSNR")
    plt.legend()
    plt.savefig(args.checkpoint_dir + "/history/Train metrics history.png")
    plt.close()
    
    plt.figure()
    plt.title("SSIM and PSNR Curve for Test Data")
    plt.xlabel("Step No.")
    plt.ylabel("Value")
    plt.plot(list(range(1, len(test_SSIM_history)+1)), test_SSIM_history, "blue", label="SSIM")
    plt.plot(list(range(1, len(test_PSNR_history)+1)), test_PSNR_history, "red", label="PSNR")
    plt.legend()
    plt.savefig(args.checkpoint_dir + "/history/Test metrics history.png")
    plt.close()
        
if __name__ == '__main__':

    parser = config.get_argparser()

    args = parser.parse_args()

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
        os.mkdir(args.checkpoint_dir + "/models")
        os.mkdir(args.checkpoint_dir + "/history")

    train(args)

