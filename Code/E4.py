# E4_vit_deep_vpt.py

from common_setup import *
from data_preprocessing import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time

class CLIPWithDeepVPT(nn.Module):
    def __init__(self, clip_model, num_classes, num_prompt_tokens=1):
        super(CLIPWithDeepVPT, self).__init__()
        self.clip = clip_model
        self.num_prompt_tokens = num_prompt_tokens
        self.num_layers = len(self.clip.vision_model.encoder.layers)

        # Freeze all CLIP model parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Initialize learnable prompt tokens for each Transformer layer
        # Shape: (num_layers, num_prompt_tokens, hidden_size)
        hidden_size = self.clip.vision_model.config.hidden_size
        self.prompt_tokens = nn.Parameter(torch.randn(self.num_layers, num_prompt_tokens, hidden_size))

        # Initialize the linear classification head
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)

    def forward(self, images):
        # Get patch embeddings
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=images, return_dict=True)
            embeddings = vision_outputs.last_hidden_state  # Shape: (batch_size, num_patches + 1, hidden_size)

        batch_size = embeddings.size(0)
        hidden_states = embeddings  # Initial input to the first layer

        # Iterate through each Transformer layer with added prompt tokens
        for layer_idx in range(self.num_layers):
            # Get the corresponding prompt tokens for this layer
            prompt = self.prompt_tokens[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_prompt_tokens, hidden_size)
            # Prepend prompt tokens to the hidden states
            modified_hidden_states = torch.cat([prompt, hidden_states], dim=1)  # Shape: (batch_size, num_patches + 1 + num_prompt_tokens, hidden_size)
            # Pass through the frozen Transformer layer
            layer = self.clip.vision_model.encoder.layers[layer_idx]
            encoder_outputs = layer(
                modified_hidden_states,      # Positional argument
                attention_mask=None,        
                output_attentions=False,
                causal_attention_mask=None  # Provide as positional argument
            )
            # **Extract the hidden_states from the tuple**
            hidden_states = encoder_outputs[0][:, self.num_prompt_tokens:, :]  # Shape: (batch_size, num_patches + 1, hidden_size)

        # Extract the [CLS] token embedding
        cls_embedding = hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Project the [CLS] embedding to the CLIP projection space
        image_features = self.clip.visual_projection(cls_embedding)  # Shape: (batch_size, projection_dim)

        # Pass through the classification head
        logits = self.classifier(image_features)  # Shape: (batch_size, num_classes)
        return logits

def main():
    # Set random seeds for reproducibility
    import random
    import numpy as np
    import torch

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(0)

    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    # Define class labels
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    num_classes = len(class_labels)

    # Initialize the Deep VPT Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_deep_vpt = CLIPWithDeepVPT(clip_model, num_classes, num_prompt_tokens=1).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only train prompt tokens and classifier
    optimizer = optim.Adam([
        {'params': model_deep_vpt.prompt_tokens, 'lr': 1e-3},
        {'params': model_deep_vpt.classifier.parameters(), 'lr': 1e-3}
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
        train_loss, train_acc = train_epoch(model_deep_vpt, train_loader, criterion, optimizer, device)
        # Evaluation
        val_loss, val_acc = evaluate(model_deep_vpt, test_loader, criterion, device)

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
            torch.save(model_deep_vpt.state_dict(), 'E4_CLIP_Deep_VPT_best_model.pth')
            print("  Best model saved!\n")

    # Plot Training Metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, "E4_CLIP_Deep_VPT")

    # Load the Best Model for Evaluation
    model_deep_vpt.load_state_dict(torch.load('E4_CLIP_Deep_VPT_best_model.pth'))

    # Detailed Evaluation
    detailed_evaluation(model_deep_vpt, test_loader, device, class_labels, "E4_CLIP_Deep_VPT")

    # Count Trainable Parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = count_trainable_parameters(model_deep_vpt)
    print(f"Number of Trainable Parameters in E4 (Deep VPT): {trainable_params}")

if __name__ == "__main__":
    main()
