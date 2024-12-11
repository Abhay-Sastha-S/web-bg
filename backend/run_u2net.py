import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
from u2net import U2NET # If you have the U2NET model in a file u2net.py

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_weights', 'u2net.pth')
model = U2NET(3,1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Preprocessing function
def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    return transform(image).unsqueeze(0)

def postprocess(pred: torch.Tensor, orig_size: (int,int)):
    # pred is a single-channel mask with values in [0,1]
    pred_img = (pred.squeeze().detach().numpy()*255).astype('uint8')
    pred_img = Image.fromarray(pred_img, mode='L')
    return pred_img.resize(orig_size, Image.BILINEAR)

def run_u2net_inference(image: Image.Image) -> Image.Image:
    orig_size = image.size
    input_tensor = preprocess(image)
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7 = model(input_tensor)
        # d1 is the output with the best resolution
        pred = d1[:,0,:,:]
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)
    mask = pred.squeeze().cpu().numpy()*255
    mask_img = Image.fromarray(mask.astype('uint8'), mode='L')
    # Apply a threshold
    # Everything above 128 is foreground
    mask_img = mask_img.point(lambda p: 255 if p > 128 else 0)
    return mask_img
