# E5_clip_dual_vpt.py

from common_setup import *
from data_preprocessing import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPModel, CLIPTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time

class CLIPWithVPT(nn.Module):
    def __init__(self, clip_model, tokenizer, num_classes, num_prompt_tokens=1):
        super(CLIPWithVPT, self).__init__()
        self.clip = clip_model
        self.tokenizer = tokenizer  # Instantiate tokenizer separately
        self.num_prompt_tokens = num_prompt_tokens

        # Freeze all CLIP model parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Initialize learnable prompt tokens for Vision backbone (Shallow VPT)
        hidden_size_vision = self.clip.vision_model.config.hidden_size
        self.vision_prompt_tokens = nn.Parameter(torch.randn(num_prompt_tokens, hidden_size_vision))

        # Initialize learnable prompt tokens for Text backbone (Shallow VPT)
        hidden_size_text = self.clip.text_model.config.hidden_size
        self.text_prompt_tokens = nn.Parameter(torch.randn(num_prompt_tokens, hidden_size_text))

        # Initialize the linear classification head
        self.classifier = nn.Linear(2 * self.clip.config.projection_dim, num_classes)

    def forward(self, images, texts=None):
        # Vision Prompt Tuning
        with torch.no_grad():
            # Get vision embeddings
            vision_outputs = self.clip.vision_model(pixel_values=images, return_dict=True)
            vision_embeddings = vision_outputs.last_hidden_state  # Shape: (batch_size, num_patches + 1, hidden_size)

        batch_size = vision_embeddings.size(0)
        # Prepend vision prompt tokens to the embeddings
        vision_prompts = self.vision_prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_prompt_tokens, hidden_size)
        modified_vision_embeddings = torch.cat([vision_prompts, vision_embeddings], dim=1)  # Shape: (batch_size, num_patches + 1 + num_prompt_tokens, hidden_size)

        # Pass through the frozen vision encoder
        vision_encoder = self.clip.vision_model.encoder
        vision_encoder_outputs = vision_encoder(
            modified_vision_embeddings,      # Positional argument
            attention_mask=None,            
            output_attentions=False,
            causal_attention_mask=None      # Provide as positional argument
        )

        # *Extract the hidden_states from the tuple*
        cls_embedding = vision_encoder_outputs[0][:, self.num_prompt_tokens, :]  # Shape: (batch_size, hidden_size)
        # Text Prompt Tuning
        if texts is not None:
            # Tokenize texts
            text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(images.device)
            with torch.no_grad():
                text_outputs = self.clip.text_model(input_ids=text_inputs["input_ids"],
                                                   attention_mask=text_inputs["attention_mask"],
                                                   return_dict=True)
                text_embeddings = text_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

            # Prepend text prompt tokens to the text embeddings
            text_prompts = self.text_prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_prompt_tokens, hidden_size)
            modified_text_embeddings = torch.cat([text_prompts, text_embeddings], dim=1)  # Shape: (batch_size, seq_len + num_prompt_tokens, hidden_size)

            # Pass through the frozen text encoder
            text_encoder = self.clip.text_model.encoder
            text_encoder_outputs = text_encoder(
                modified_text_embeddings,      # Positional argument
                attention_mask=None,           
                output_attentions=False,
                causal_attention_mask=None      # Provide as positional argument
            )

            # **Extract the hidden_states from the tuple**
            text_cls_embedding = text_encoder_outputs[0][:, -1, :]  # Shape: (batch_size, hidden_size)

            # Project the text embeddings
            text_features = self.clip.text_projection(text_cls_embedding)  # Shape: (batch_size, projection_dim)
        else:
            text_features = None

        
        # Feature Combination
        # Project vision embeddings
        vision_features = self.clip.visual_projection(cls_embedding)  # Shape: (batch_size, projection_dim)

        if text_features is not None:
            # Combine vision and text features 
            image_features = torch.cat([vision_features, text_features], dim=1) #Shape: (batch_size, 2*projection_dim)
            
        else:
            image_features = torch.cat([vision_features, vision_features], dim=1) #Shape: (batch_size, 2*projection_dim)

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

    # Initialize the CLIP model and tokenizer
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    # Define class labels
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    num_classes = len(class_labels)

    # Initialize the VPT Model (Shallow VPT on both Vision and Text)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_vpt = CLIPWithVPT(clip_model, tokenizer, num_classes, num_prompt_tokens=1).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only train vision prompts, text prompts, and classifier
    optimizer = optim.Adam([
        {'params': model_vpt.vision_prompt_tokens, 'lr': 1e-3},
        {'params': model_vpt.text_prompt_tokens, 'lr': 1e-3},
        {'params': model_vpt.classifier.parameters(), 'lr': 1e-3}
    ])

    # Training Loop
    epochs = 3
    best_val_acc = 0.0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        model_vpt.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            texts = [class_labels[label] for label in labels.cpu().numpy()]  # Assuming labels are indices

            optimizer.zero_grad()
            outputs = model_vpt(images, texts=texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluation
        val_loss, val_acc = evaluate(model_vpt, test_loader, criterion, device, use_text=True)

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
            torch.save(model_vpt.state_dict(), 'E5_CLIP_VPT_best_model.pth')
            print("  Best model saved!\n")

    # Plot Training Metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, "E5_CLIP_VPT")

    # Load the Best Model for Evaluation
    model_vpt.load_state_dict(torch.load('E5_CLIP_VPT_best_model.pth'))

    # Detailed Evaluation
    detailed_evaluation(model_vpt, test_loader, device, class_labels, "E5_CLIP_VPT", use_text=True)

    # Count Trainable Parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = count_trainable_parameters(model_vpt)
    print(f"Number of Trainable Parameters in E5 (CLIP VPT): {trainable_params}")

if __name__ == "__main__":
    main()
