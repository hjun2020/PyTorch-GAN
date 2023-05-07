from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=False, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Prepare input
image_tensor = Variable(transform(Image.open(opt.image_path))).to(device).unsqueeze(0)

# Open the input video file
input_video = cv2.VideoCapture('/home/ubuntu/PyTorch-GAN/implementations/esrgan/cpp/people_crossing_352x240.mp4')

# Get the video properties
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(input_video.get(cv2.CAP_PROP_FPS))

# Create the output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('./videos/output_video.mp4', fourcc, frame_rate, (frame_width, frame_height))


# ret, frame = input_video.read()


# # Convert the frame to a tensor
# tensor_frame = transform(frame).to(device).unsqueeze(0)
# with torch.no_grad():
#     sr_image = denormalize(generator(tensor_frame)).cpu()

# save_image(sr_image, "./videos/temp.png")

count = 0
while True:
    # Read the next frame
    count += 1
    if count % 1 == 0:
        print(count)
    if count > 30:
        break
    ret, frame = input_video.read()

    
    # If no frame is read, break out of the loop
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = transform(frame).to(device).unsqueeze(0)
 
    # Pass the frame through the ESRGAN model
    with torch.no_grad():
        output_tensor = denormalize(generator(tensor_frame)).cpu()
    
    # print(output_tensor.shape)
    save_image(output_tensor, "./videos/temp.png")
    # Convert the output tensor to a numpy array and resize it to the original frame size
    sr_image_np = output_tensor.squeeze().numpy().transpose(1, 2, 0)
    sr_image_np = cv2.resize(sr_image_np, (1280, 720), interpolation=cv2.INTER_CUBIC)
    sr_image_np = (sr_image_np).clip(0, 255).astype('uint8')

    # Write frame to video file
    output_video.write(sr_image_np)
    
input_video.release()
output_video.release()