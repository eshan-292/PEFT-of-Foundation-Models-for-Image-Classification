# E6_full_finetune_vit.py

from common_setup import *
from data_preprocessing import *
from utils import *
from transformers import CLIPModel, CLIPProcessor

class CLIPWithFullFinetune(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPWithFullFinetune, self).__init__()
        self.clip = clip_model
        
        # Unfreeze all parameters of the vision backbone
        for param in self.clip.vision_model.parameters():
            param.requires_grad = True
        
        # Initialize the linear classification head
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)
        
    def forward(self, images):
        # Get image features
        image_features = self.clip.get_image_features(images)  # Shape: (batch_size, 512)
        
        # Pass through the classifier
        logits = self.classifier(image_features)
        return logits

def main():

    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Initialize the Full Fine-Tuning Model
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_full_finetune = CLIPWithFullFinetune(clip_model, num_classes).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimize all vision_model parameters and the classifier
    optimizer = optim.Adam([
        {'params': model_full_finetune.clip.vision_model.parameters(), 'lr': 1e-5},
        {'params': model_full_finetune.classifier.parameters(), 'lr': 1e-3}
    ])

    # Training Loop
    epochs = 500
    best_val_acc = 0.0
    patience = 5
    trigger_times = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model_full_finetune, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model_full_finetune, test_loader, criterion, device)
        
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
            torch.save(model_full_finetune.state_dict(), 'best_model_full_finetune.pth')
            trigger_times = 0
            print("  Best model saved!\n")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered!")
                break

    # Plot Training Metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, "E6_Full_Finetune_ViT")

    # Load the Best Model for Evaluation
    model_full_finetune.load_state_dict(torch.load('best_model_full_finetune.pth'))

    # Detailed Evaluation
    detailed_evaluation(model_full_finetune, test_loader, device, labels, "E6_Full_Finetune_ViT")

    # Count Trainable Parameters
    def count_trainable_parameters_full_finetune(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of Trainable Parameters in Full Fine-Tuning: {count_trainable_parameters_full_finetune(model_full_finetune)}")


if __name__ == "__main__":
    main()