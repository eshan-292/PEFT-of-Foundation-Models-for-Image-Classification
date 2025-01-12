# E1_clip_zero_shot.py

from common_setup import *
from data_preprocessing import *
from utils import *

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class CLIPZeroShotClassifier:
    def __init__(self, clip_model, class_labels, device):
        """
        Initializes the CLIP Zero-Shot Classifier.

        Args:
            clip_model (CLIPModel): Pre-trained CLIP model.
            class_labels (list): List of class label strings.
            device (torch.device): Device to run the model on.
        """
        self.clip = clip_model
        self.class_labels = class_labels
        self.device = device
        
        # # requires grad is set to False for all parameters
        # for param in self.clip.parameters():
        #     param.requires_grad = False

        # Move the model to the specified device and set to evaluation mode
        self.clip.to(self.device)
        self.clip.eval()

        # Define text prompts for each class
        self.text_prompts = [f"A photo of a {label}" for label in self.class_labels]

        # Initialize the CLIP processor for tokenization
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # Tokenize and encode the text prompts
        self.text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.text_embeddings = self.clip.get_text_features(**self.text_inputs)
            # Normalize the text embeddings
            self.text_embeddings /= self.text_embeddings.norm(dim=-1, keepdim=True)

    def predict(self, images):
        """
        Predicts class labels for a batch of images using zero-shot inference.

        Args:
            images (torch.Tensor): Batch of input images. Shape: (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Predicted class indices. Shape: (batch_size,)
        """
        with torch.no_grad():
            # Encode the images to obtain image embeddings
            image_embeddings = self.clip.get_image_features(images)
            # Normalize the image embeddings
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            # Compute cosine similarity between image and text embeddings
            similarity = image_embeddings @ self.text_embeddings.t()  # Shape: (batch_size, num_classes)
            # Determine the predicted class by finding the index with the highest similarity
            preds = similarity.argmax(dim=-1)
        return preds

def main():
    # Initialize the CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    # Define class labels
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

    # Initialize the Zero-Shot Classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    zero_shot_classifier = CLIPZeroShotClassifier(clip_model, class_labels, device)

    # Initialize lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Iterate through the test dataset
    for batch in test_loader:
        images = batch['image'].to(device)  # Shape: (batch_size, 3, 224, 224)
        labels = batch['label'].to(device)  # Shape: (batch_size,)

        # Get predictions
        preds = zero_shot_classifier.predict(images)  # Shape: (batch_size,)

        # Append to the lists
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays for evaluation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Generate Classification Report
    print("\n=== Classification Report for E1_CLIP_Zero_Shot ===")
    print(classification_report(all_labels, all_preds, target_names=class_labels))

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - E1_CLIP_Zero_Shot')
    plt.savefig('E1_CLIP_Zero_Shot_confusion_matrix.png')
    plt.show()

    # Count Trainable Parameters (Should be zero)
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Since this is a zero-shot setup, all model parameters should be frozen
    trainable_params = count_trainable_parameters(zero_shot_classifier.clip)

    print(f"Number of Trainable Parameters in E1 (Zero-Shot): {trainable_params}")

if __name__ == "__main__":
    main()
