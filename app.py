import torch
import gradio as gr
import torchvision
import numpy as np
import random
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the pre-trained model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Function to process image and return bounding boxes and count
# Threshold for confidence scores is set to a default value of 0.8
def process_image(image, threshold=0.8):
    # Convert the image to a tensor
    image_tensor = F.to_tensor(image)
    
    # Run the model
    with torch.no_grad():
        output = model([image_tensor])[0]
    
    # Convert the original image to RGB
    image = image.convert("RGB")
    
    # Draw bounding boxes and masks on the image
    draw = ImageDraw.Draw(image)
    boxes = output['boxes']
    scores = output['scores']
    masks = output['masks']
    
    for idx, score in enumerate(scores):
        if score >= threshold:
            box = boxes[idx].detach().numpy()
            mask = masks[idx, 0].detach().numpy()
            color = (int(random.random() * 255), int(random.random() * 255), int(random.random() * 255))
            
            # Draw bounding box
            x0, y0, x1, y1 = box
            draw.rectangle(((x0, y0), (x1, y1)), outline=color, width=3)
            
            # Overlay the mask
            mask_image = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            mask_image = mask_image.resize(image.size)
            mask_overlay = Image.new("RGB", image.size, color)
            image.paste(mask_overlay, (0, 0), mask_image)
    
    return image


# Gradio interface function
def gradio_interface(image):
    image_with_boxes = process_image(image)
    return image_with_boxes


# Set up Gradio Interface with the new API
demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Object segmentation with Mask R-CNN",
    description="Upload an image for object segmentation.",
)

# Launch the Gradio app
demo.launch(share=True)
