# data_preprocessing.py

from common_setup import *

# Define Transformations Including Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# Load the Dataset
dataset = load_dataset("aggr8/brain_mri_train_test_split")

# Define Class Labels and Optional Index-to-Label Mapping
labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
# Since labels are already integers, no need for label_to_idx
idx_to_label = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}

# Preprocessing Function Without Mapping
def preprocess(example):
    image = example['image'].convert('RGB')  # Directly convert without Image.open
    image = transform(image)
    label = example['label']  # Use the integer label directly
    return {'image': image, 'label': label}

# Apply Preprocessing to Train and Test Splits
train_dataset = dataset['train'].map(preprocess, remove_columns=dataset['train'].column_names)
test_dataset = dataset['test'].map(preprocess, remove_columns=dataset['test'].column_names)

# Set the Format for PyTorch
train_dataset.set_format(type='torch', columns=['image', 'label'])
test_dataset.set_format(type='torch', columns=['image', 'label'])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


