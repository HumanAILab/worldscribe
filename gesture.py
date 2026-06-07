import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import firebase_admin
from firebase_admin import credentials
from utils.env_config import FIREBASE_CREDENTIALS, FIREBASE_DATABASE_URL
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DATABASE_URL
    })
from utils.gesture.fullmodel import ObjectInHand

class HandObjectCropper:
    def __init__(self, object_model, human_model_checkpoint, object_model_checkpoint):
        """
        Initializes the pipeline with the provided models and checkpoints.
        """
        self.model = object_model
        self.model.load_model(human_model_checkpoint, object_model_checkpoint)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses the image for the model.
        """
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0)  
        return image.to(self.device)

    def postprocess_masks(self, object_mask, hand_mask):
        """
        Post-processes the predicted masks to crop the object.
        """
        object_mask_np = object_mask.squeeze().cpu().numpy()
        hand_mask_np = hand_mask.squeeze().cpu().numpy()
        # TODO: implement a check to see if hands or object detected 
        # TODO: implement a threshold to filter out the hand 
        # if information is not sufficient, use mediapipe
        combined_mask = (object_mask_np > 0.5) & (hand_mask_np > 0.5)
        return combined_mask

    def crop_object(self, image, mask):
        """
        Crops the region containing the object based on the mask.
        """
        np_image = np.array(image)
        y_indices, x_indices = np.where(mask)

        if y_indices.size == 0 or x_indices.size == 0:
            raise ValueError("No object detected in the hand.")

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        cropped_image = np_image[y_min:y_max, x_min:x_max]
        return Image.fromarray(cropped_image)

    def process_image(self, image_path):
        """
        Processes an input image to identify and crop the object in the user's hand.
        """
        original_image = Image.open(image_path).convert("RGB")
        input_image = self.preprocess_image(image_path)

        with torch.no_grad():
            object_mask, hand_mask = self.model(input_image)

        combined_mask = self.postprocess_masks(object_mask, hand_mask)

        cropped_image = self.crop_object(original_image, combined_mask)

        return cropped_image

if __name__ == "__main__":
    human_model_checkpoint = "./models/resnet18_adam.pth.tar"
    object_model_checkpoint = "./models/unet-b0-bgr-100epoch.pt"

    model = ObjectInHand(arch="resnet18", num_classes=20)

    cropper = HandObjectCropper(model, human_model_checkpoint, object_model_checkpoint)

    image_path = "./test.png"

    try:
        cropped_object = cropper.process_image(image_path)
        cropped_object.save("cropped_object.png")
        print("Cropped object saved as 'cropped_object.png'.")
    except ValueError as e:
        print(str(e))