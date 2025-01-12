# E2_clip_linear_probing.py

from common_setup import *
from data_preprocessing import *
from utils import *
import torch.nn as nn
import torch.optim as optim

class CLIPWithLinearClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPWithLinearClassifier, self).__init__()
        self.clip = clip_model

        # Freeze all CLIP model parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Initialize the linear classification head
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)
        
    def forward(self, images):
        # Get image features (512-dimensional)
        image_features = self.clip.get_image_features(images)  # Shape: (batch_size, 512)
        
        # Pass image_features through the classifier
        logits = self.classifier(image_features)  # Shape: (batch_size, num_classes)
        return logits

def main():
    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    print(clip_model)
    exit()
    
    # Initialize the Linear Probing Model
    num_classes = 4  # Adjust based on your dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_linear = CLIPWithLinearClassifier(clip_model, num_classes).to(device)
    
    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_linear.classifier.parameters(), lr=1e-3)
    
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
        train_loss, train_acc = train_epoch(model_linear, train_loader, criterion, optimizer, device)
        
        # Evaluation
        val_loss, val_acc = evaluate(model_linear, test_loader, criterion, device)
        
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
            torch.save(model_linear.state_dict(), 'best_model_linear_probing.pth')
            print("  Best model saved!\n")
    
    # Plot Training Metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, "E2_CLIP_Linear_Probing")
    
    # Load the Best Model for Evaluation
    model_linear.load_state_dict(torch.load('best_model_linear_probing.pth'))
    
    # Detailed Evaluation
    labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']  # Adjust based on your dataset
    detailed_evaluation(model_linear, test_loader, device, labels, "E2_CLIP_Linear_Probing")
    
    # Count Trainable Parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Number of Trainable Parameters in E2: {count_trainable_parameters(model_linear)}")

if __name__ == "__main__":
    main()
