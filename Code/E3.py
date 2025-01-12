# E3_vit_shallow_vpt.py

from common_setup import *
from data_preprocessing import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import time

class CLIPWithShallowVPT(nn.Module):
    def __init__(self, clip_model, num_classes, num_prompt_tokens=1):
        super(CLIPWithShallowVPT, self).__init__()
        self.clip = clip_model
        self.num_prompt_tokens = num_prompt_tokens

        # Freeze all CLIP model parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Initialize learnable prompt tokens for shallow VPT
        # Shape: (num_prompt_tokens, hidden_size)
        hidden_size = self.clip.vision_model.config.hidden_size
        self.prompt_tokens = nn.Parameter(torch.randn(num_prompt_tokens, hidden_size))

        # Initialize the linear classification head
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)

    def forward(self, images):
        # Get patch embeddings
        with torch.no_grad():
            # Obtain the vision outputs; replace 'embeddings' with 'last_hidden_state'
            vision_outputs = self.clip.vision_model(pixel_values=images, return_dict=True)
            embeddings = vision_outputs.last_hidden_state  # Shape: (batch_size, num_patches + 1, hidden_size)

        batch_size = embeddings.size(0)
        # Prepend prompt tokens to the embeddings of the first layer
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_prompt_tokens, hidden_size)
        # Concatenate prompt tokens with embeddings
        # Assuming that [CLS] token is the first token
        # New embeddings shape: (batch_size, num_patches + 1 + num_prompt_tokens, hidden_size)
        modified_embeddings = torch.cat([prompt_tokens, embeddings], dim=1)

        # Pass the modified embeddings through the frozen vision encoder
        # Access the encoder layers; CLIPVisionModel has an 'encoder' attribute
        vision_encoder = self.clip.vision_model.encoder

        # Forward the modified embeddings through the encoder
        # Replace 'hidden_states' with 'inputs_embeds' and remove 'head_mask'
        encoder_outputs = vision_encoder(
            inputs_embeds=modified_embeddings,
            attention_mask=None,  
            # head_mask=None,  # Removed as it's not accepted
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        # Extract the [CLS] token embedding (assumed to be at position num_prompt_tokens)
        cls_embedding = encoder_outputs.last_hidden_state[:, self.num_prompt_tokens, :]  # Shape: (batch_size, hidden_size)

        # Project the [CLS] embedding to the CLIP projection space
        image_features = self.clip.visual_projection(cls_embedding)  # Shape: (batch_size, projection_dim)

        # Pass through the classification head
        logits = self.classifier(image_features)  # Shape: (batch_size, num_classes)
        return logits

def main():
    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    # Define class labels
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    num_classes = len(class_labels)

    # Initialize the Shallow VPT Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_shallow_vpt = CLIPWithShallowVPT(clip_model, num_classes, num_prompt_tokens=1).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only train prompt tokens and classifier
    optimizer = optim.Adam([
        {'params': model_shallow_vpt.prompt_tokens, 'lr': 1e-3},
        {'params': model_shallow_vpt.classifier.parameters(), 'lr': 1e-3}
    ])

    # Training Loop
    epochs = 500
    best_val_acc = 0.0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Training
        train_loss, train_acc = train_epoch(model_shallow_vpt, train_loader, criterion, optimizer, device)
        # Evaluation
        val_loss, val_acc = evaluate(model_shallow_vpt, test_loader, criterion, device)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch {epoch}/{epochs} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%\n")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model_shallow_vpt.state_dict(), 'E3_CLIP_Shallow_VPT_best_model.pth')
            print("  Best model saved!\n")

    # Plot Training Metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, "E3_CLIP_Shallow_VPT")

    # Load the Best Model for Evaluation
    model_shallow_vpt.load_state_dict(torch.load('E3_CLIP_Shallow_VPT_best_model.pth'))

    # Detailed Evaluation
    detailed_evaluation(model_shallow_vpt, test_loader, device, class_labels, "E3_CLIP_Shallow_VPT")

    # Count Trainable Parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = count_trainable_parameters(model_shallow_vpt)
    print(f"Number of Trainable Parameters in E3 (Shallow VPT): {trainable_params}")

if __name__ == "__main__":
    main()
