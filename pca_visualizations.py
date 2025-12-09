
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import json
import os

# CONFIGURATION
DATA_DIR = "animal data"
CLASSES = ["mammal", "reptile", "bird", "fish", "amphibian", "insect"]
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET CLASS
class AnimalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# HELPER FUNCTIONS
def load_dataset(data_dir):
    """Load all images and their labels"""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    return image_paths, labels

def get_model(num_classes=6):
    """Load model architecture"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

# VISUALIZATION 1: f(θ) - Weight Evolution
def visualize_weight_evolution():
    """
    Creates 3D plot showing how model weights evolved during training.
    X, Y = PCA components of weights
    Z = Loss value at that epoch
    """
    print("Creating Visualization 1: Model Weight Evolution f(θ)")
    
    # Load training history
    if not os.path.exists('weight_history.npy'):
        print("ERROR: weight_history.npy not found!")
        print("You need to run the modified training script first.")
        return
    
    weight_history = np.load('weight_history.npy', allow_pickle=True)
    
    # Load loss history
    with open('training_results.json', 'r') as f:
        results = json.load(f)
    train_losses = results['history']['train_loss']
    
    print(f"Loaded {len(weight_history)} weight snapshots")
    print(f"Weight vector dimension: {weight_history[0].shape[0]:,} parameters")
    
    # Apply PCA to reduce to 2D
    print("Applying PCA to weight vectors...")
    pca = PCA(n_components=2)
    weights_2d = pca.fit_transform(weight_history)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    x = weights_2d[:, 0]
    y = weights_2d[:, 1]
    z = train_losses
    
    # Color points by epoch
    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
    
    # Plot points
    scatter = ax.scatter(x, y, z, c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Connect points with lines to show trajectory
    ax.plot(x, y, z, 'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=300, marker='*', 
               edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=300, marker='*', 
               edgecolors='black', linewidth=2, label='End', zorder=10)
    
    # Labels and title
    ax.set_xlabel('PC1 (Weight Space)', fontsize=12, labelpad=10)
    ax.set_ylabel('PC2 (Weight Space)', fontsize=12, labelpad=10)
    ax.set_zlabel('Training Loss', fontsize=12, labelpad=10)
    ax.set_title('Model Weight Evolution During Training f(θ)', fontsize=14, pad=20)
    ax.legend(fontsize=10)
    
    # Add epoch annotations
    step = max(1, len(x) // 5)
    for i in range(0, len(x), step):
        ax.text(x[i], y[i], z[i], f'  E{i}', fontsize=8, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('pca_weight_evolution.png', dpi=300, bbox_inches='tight')
    print("Saved: pca_weight_evolution.png")
    plt.close()

# VISUALIZATION 2: g(x) - Prediction Surface
def visualize_prediction_surface():
    print("\nCreating Visualization 2: Prediction Surface g(x)")
    
    # Load best model
    if not os.path.exists('best_model.pth'):
        print("ERROR: best_model.pth not found!")
        print("You need to train the model first.")
        return
    
    model = get_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test images
    print("Loading test images...")
    image_paths, labels = load_dataset(DATA_DIR)
    
    # Sample ~200 images
    num_samples = min(200, len(image_paths))
    indices = np.random.choice(len(image_paths), num_samples, replace=False)
    sampled_paths = [image_paths[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    
    print(f"Using {num_samples} images for visualization")
    
    # Transform for loading images
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and flatten all images for PCA
    print("Loading and flattening images...")
    flattened_images = []
    predictions = []
    true_labels = []
    
    for img_path, true_label in zip(sampled_paths, sampled_labels):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred_class = torch.argmax(output, dim=1).item()
            
            # Flatten image for PCA
            img_raw = Image.open(img_path).convert('RGB')
            img_raw = img_raw.resize((IMG_SIZE, IMG_SIZE))
            img_flat = np.array(img_raw).flatten()
            
            flattened_images.append(img_flat)
            predictions.append(pred_class)
            true_labels.append(true_label)
        except:
            continue
    
    flattened_images = np.array(flattened_images)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    print(f"Successfully processed {len(flattened_images)} images")
    print(f"Image vector dimension: {flattened_images[0].shape[0]:,} pixels")
    
    # Apply PCA to reduce to 2D
    print("Applying PCA to image vectors...")
    pca = PCA(n_components=2)
    images_2d = pca.fit_transform(flattened_images)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by predicted class
    x = images_2d[:, 0]
    y = images_2d[:, 1]
    z = predictions
    
    # Create color map for classes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Plot each class separately for legend
    for class_idx, class_name in enumerate(CLASSES):
        mask = predictions == class_idx
        if np.any(mask):
            ax.scatter(x[mask], y[mask], z[mask], 
                      c=colors[class_idx], 
                      label=class_name, 
                      s=50, 
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=0.5)
    
    # Labels and title
    ax.set_xlabel('PC1 (Input Space)', fontsize=12, labelpad=10)
    ax.set_ylabel('PC2 (Input Space)', fontsize=12, labelpad=10)
    ax.set_zlabel('Predicted Class', fontsize=12, labelpad=10)
    ax.set_title('Prediction Surface g(x) in Input Space', fontsize=14, pad=20)
    
    # Set z-axis to show class names
    ax.set_zticks(range(len(CLASSES)))
    ax.set_zticklabels(CLASSES, fontsize=8)
    
    ax.legend(fontsize=9, loc='upper left')
    
    # Calculate accuracy
    accuracy = np.mean(predictions == true_labels)
    ax.text2D(0.02, 0.98, f'Accuracy: {accuracy:.2%}', 
             transform=ax.transAxes, fontsize=12, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pca_prediction_surface.png', dpi=300, bbox_inches='tight')
    print("Saved: pca_prediction_surface.png")
    plt.close()

# MAIN
def main():
    print("\nPCA VISUALIZATION GENERATOR\n")
    
    # Check if training artifacts exist
    if not os.path.exists('best_model.pth'):
        print("WARNING: best_model.pth not found!")
        print("Please run the training script first.\n")
        return
    
    # Create both visualizations
    visualize_weight_evolution()
    visualize_prediction_surface()
    
    print("\nAll visualizations complete!")
    print("\nGenerated files:")
    print("  1. pca_weight_evolution.png")
    print("  2. pca_prediction_surface.png")

if __name__ == "__main__":
    main()