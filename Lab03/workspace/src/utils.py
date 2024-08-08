import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from models.resnet34_unet import *
from models.unet import *
import os
import numpy as np
import matplotlib.pyplot as plt


def dice_score(pred_mask, gt_mask):
    # implement the Dice score here

    # Dice score = 2 * (number of common pixels) / (predicted img size + groud truth img size)
    # [batch_size,1,256,256]
    assert pred_mask.shape == gt_mask.shape
    assert pred_mask.shape[2] == 256
    assert pred_mask.shape[3] == 256
    pred_mask = (pred_mask > 0.5) # return 1,0
    gt_mask = (gt_mask) # return 1,0

    
    intersection = (pred_mask == gt_mask).sum(dim=(1, 2, 3))
    
    #pred_img_sz = pred_mask.sum(dim=(1, 2, 3))
    #ground_truth_img_sz = gt_mask.sum(dim=(1, 2, 3))
    score = (2. * intersection) / (pred_mask.shape[2]*pred_mask.shape[3]*2)

    return score.mean()

def get_model(model_name,model_path,device):
    if (model_name == "unet"):
        model = UNet(input_channel=3,output_channel=1).to(device) # binary semantic segmentation,
    else:
        model = ResNet34_UNet(input_channel=3,output_channel=1).to(device) 
        model_name = "resnet_unet"
        pass

    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.99)

    torch.cuda.empty_cache()
    # Load the model state
    if os.path.isfile(model_path):
        state = torch.load(model_path)
        model.load_state_dict(state['model_state'])
        last_epoch = state['epoch']
        last_loss = state['loss']
        try:
            score = state['score']
        except:
            score = "NAN"
        print(f"Checkpoint loaded from {model_path} , last_epoch: {last_epoch} , last_loss: {last_loss} , dice_score: {score}")
    else:
        print(f"No checkpoint file found at {model_path}")
        
    return model

def plot_line_chart(x, y, title='Line Chart', x_label='X-axis', y_label='Y-axis'):

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data')
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

def show_batch_masked_img(images,pred_masks):
    # images [batch,3,256,256]
    # pred_masks [batch,1,256,256]
    # Ensure inputs are on CPU and convert to numpy
    images = images.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()

    # Normalize images to [0, 1]
    images = images / 255.0

    # List to hold masked images
    masked_images = []

    # Draw masks on each image
    for i in range(images.shape[0]):
        masked_image = utils.draw_segmentation_masks(
            image=torch.tensor(images[i]),  # Convert numpy array back to tensor
            masks=torch.tensor(pred_masks[i]),  # Convert to boolean tensor
            alpha=0.5,  # Transparency level
            colors=["red"]
        )
        masked_images.append(masked_image.permute(1, 2, 0).numpy())

    # Plot images and masks
    num_images = images.shape[0]
    plt.figure(figsize=(16, 8 * num_images))

    for i in range(num_images):
        # Display original image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(images[i].transpose(1, 2, 0))  # Convert [C, H, W] to [H, W, C]
        plt.axis('off')
        plt.title(f"Original Image {i + 1}")

        # Display masked image
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(masked_images[i])
        plt.axis('off')
        plt.title(f"Masked Image {i + 1}")

    plt.tight_layout()
    plt.show()



def show_masked_img(model_path,model_name,image_path):
    image = np.array(Image.open(image_path).convert("RGB"),dtype='uint8')
    original_image = Image.fromarray(image)
    image = np.array(Image.fromarray(image).resize((256, 256), Image.BILINEAR))
    image = np.moveaxis(image, -1, 0)
    image = np.expand_dims(image, axis=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(model_name,model_path,device)
    
    model.eval()
    tensor_image = torch.Tensor(image)
    tensor_image = tensor_image.to(device)

    outputs = model(tensor_image)

    pred_mask = F.sigmoid(outputs)

    pred_mask = (pred_mask > 0.5) # predict result
    
    
    # Draw masks on the image
    # Remove the batch dimension and move tensors to CPU
    tensor_image = tensor_image.squeeze().cpu()
    pred_mask = pred_mask.squeeze().cpu()

    # Use torchvision.utils.draw_segmentation_masks to overlay masks
    # Note: `masks` should be in the shape (N, H, W) where N is the number of masks
    pred_mask = pred_mask.bool()
    print(pred_mask.dtype)

    tensor_image = tensor_image / 255.0

    masked_image = utils.draw_segmentation_masks(
        image=tensor_image,
        masks=pred_mask,
        alpha=0.5,  # Transparency level
        colors=["red"]
    )

    # Convert the masked image to a numpy array
    masked_image_np = masked_image.permute(1, 2, 0).numpy()

    # Convert original image to numpy array
    original_image_np = np.array(original_image.resize((256, 256), Image.BILINEAR))

    # Plot both images side by side
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np)
    plt.axis('off')
    plt.title(f"{model_name} | Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(masked_image_np)
    plt.axis('off')
    plt.title(f"{model_name} | Image with Segmentation Masks")
    
    plt.show()