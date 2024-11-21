import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--code_dir", type=str, default="./")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="./data", 
                        help="path to the dataset csv file")
    parser.add_argument("--data_dir", type=str, default="/content")
    parser.add_argument("--trained_model", type=str, default="./checkpoints/models/best_model.pt")

    # Train
    parser.add_argument("--use_manual_seed", type=bool, default=False, help="Whether use manual seed or not")
    parser.add_argument("--seed", type=int, default=72322)
    parser.add_argument("--image_size", type=int, nargs="+", default=256,
                        help="Size of the input image to the model. It should be entered like two seperated integer number. Default=256x256")
    parser.add_argument("--critic_update", type=int, default=5, 
                        help="Number of iterations of critical model in each batch iteration")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=5, 
                        help="Saving the model weights and loss/metric plots after every these many steps")
    parser.add_argument("--load_checkpoint", type=str, default="./checkpoints/models/last_model.pt",
                        help="Start from the last checkpoint. Always refer as last_model.pt")
    parser.add_argument("--c_lambda", type=float, default=10,
                        help="indicate the amount of the gradient penalty in critic loss")
    parser.add_argument("--percept_weight", type=float, default=100,
                        help="Weight of the perceptual loss in the calculation of the generator model loss")
    
    # Optimizer and scheduler
    parser.add_argument("--init_lr", type=float, default=1e-2,
                        help="Initial learning rate for scheduler")
    parser.add_argument("--final_lr", type=float, default=1e-6,
                        help="Final learning rate for scheduler")
    parser.add_argument("--weight_decay_factor", type=float, default=0.2,
                        help="Weight decay factor for scheduler")
    parser.add_argument("--lr_wait", type=int, default=3,
                        help="Number of cicles without significant improvement in loss for scheduler")
    parser.add_argument("--lr_thresh", type=float, default=1e-3,
                        help="Threshold to check plateau-ing of loss")
    parser.add_argument("--momentum1", type=float, default=0.5,
                        help="Optimizer momentum 1 value")
    parser.add_argument("--momentum2", type=float, default=0.999,
                        help="Optimizer momentum 1 value")
    
    return parser
    
