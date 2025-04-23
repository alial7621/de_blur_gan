import torch
from torchvision import transforms
from PIL import Image
import argparse
from deblur_modules.models import Generator

def load_model(checkpoint_path, device):
    model = Generator()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Deblurring')
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    
    result = predict(args.image_path, './checkpoints/models/best_model.pt')

    result_img = transforms.ToPILImage()(result[0].cpu())
    img = Image.open(args.image_path)
    width, height = img.size
    new_im = Image.new('RGB', (width*2, height))
    new_im.paste(img, (0,0))
    new_im.paste(result_img, (width,0))
    new_im.show()
    
    print("Prediction completed.")