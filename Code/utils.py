# utils.py

from common_setup import *
from data_preprocessing import *

# Training Function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



import torch

def evaluate(model, data_loader, criterion, device, use_text=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            

            if use_text:
                # texts = batch['text']  # Ensure 'text' is available in the batch
                texts = [class_labels[label] for label in labels.cpu().numpy()]  # Assuming labels are indices
                outputs = model(images, texts=texts)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Plot Metrics Function
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, experiment_name):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{experiment_name} - Loss Curves')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), [acc * 100 for acc in train_accuracies], label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), [acc * 100 for acc in val_accuracies], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{experiment_name} - Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{experiment_name}_training_curves.png')
    plt.show()

# Detailed Evaluation Function
def detailed_evaluation(model, loader, device, labels, experiment_name ,useText = False):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels_batch = batch['label'].to(device)

            if useText:
                # texts = batch['text']  # Ensure 'text' is available in the batch
                texts = [labels[label] for label in labels_batch.cpu().numpy()]
                outputs = model(images, texts=texts)
            else:
                outputs = model(images)            
            # outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    # Classification Report
    print(f"Classification Report for {experiment_name}:")
    print(classification_report(all_labels, all_preds, target_names=labels))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{experiment_name} - Confusion Matrix')
    plt.savefig(f'{experiment_name}_confusion_matrix.png')
    plt.show()
