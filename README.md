# Object Segmentation with Mask R-CNN
This is a simple demo that shows how Mask R-CNN works on images. It is adapted from Nibra's demo.

## Project Overview
- **Model**: Pre-trained Mask R-CNN model from torchvision.
- **Features**:
  - Providing a mask on the detected Lego peices.

## Project Structure
```bash
lego-detection-maskrcnn/
├── app.py                    # Main Gradio application script
```

## Running the Application
Launch the Gradio application using app.py:

python app.py

This will start a Gradio interface at a local address, where you can:
Upload Image: Select an image containing objects.
View Results: The app displays:
Bounding boxes around each detected object.
Masks to highlight detected objects.
