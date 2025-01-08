import matplotlib.pyplot as plt
import os
def plot_loss_history(loss_history_dict, output_folder, key):
    """
    Plots the training and validation loss over epochs.

    Args:
        loss_history_dict (dict): A dictionary with keys like 'train' and 'val',
                                  where each key contains a list of loss values.
                                  Example:
                                  {
                                      "train": [0.9, 0.8, 0.7],
                                      "val": [1.0, 0.9, 0.85]
                                  }

    Returns:
        None
    """
    # Check if the dictionary contains both 'train' and 'val' keys
    if not all(key in loss_history_dict for key in ['train', 'val']):
        raise ValueError("The dictionary must contain 'train' and 'val' keys.")

    # Extract loss histories
    train_loss = loss_history_dict['train']
    val_loss = loss_history_dict['val']

    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label=f'Train {key}', marker='o')
    plt.plot(val_loss, label=f'Validation {key}', marker='o')
    
    # Add labels, title, and legend
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(key, fontsize=12)
    plt.title(f'Training and Validation {key} over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Display the plot
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{key}_plot.png')
    
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to free memory

    print(f"Loss plot saved to {output_path}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from constants import label_mapping

def plot_confusion_matrix_with_counts(data_loader, true_labels, pred_labels, task, mode):
    """
    Plots a confusion matrix with counts and normalized percentages for a multi-class classification task.
    Supports an optional mapping dictionary for class labels.

    Args:
        data_loader (DataLoader): PyTorch DataLoader used for the task.
        true_labels (list): True labels of the data.
        pred_labels (list): Predicted labels of the data.
        task (str): Path or name of the task folder for saving the plot.
        mode (str): Mode (e.g., 'train', 'test', 'val') for labeling the plot file.
        label_mapping (dict): Optional dictionary to map class indices to custom labels.

    Returns:
        None
    """
    # Access class names from the DataLoader
    dataset = data_loader.dataset
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    else:
        raise AttributeError("The dataset does not have a 'classes' attribute.")

    # Apply label mapping if provided
    if label_mapping:
        class_names = [label_mapping.get(cls, cls) for cls in class_names]

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle NaNs for empty classes

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))  # Adjust size for readability
    sns.heatmap(
        cm_normalized,
        annot=False,  # Turn off default annotations to customize them
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Value (%)'},
        linewidths=0.5,
        ax=ax
    )
    fontsize = max(3, 15 - len(class_names) // 2)  # Smaller font for more classes
    print(fontsize)
    # Add custom annotations: "count (percentage)"
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            label = f"{count} ({percentage:.1f}%)"
            ax.text(
                j + 0.5, i + 0.5,
                label,
                ha='center',
                va='center',
                color='black',
                fontsize=fontsize,
                rotation=45, 
            )

    # Customize plot
    ax.set_title(f"Confusion Matrix ({mode.capitalize()})", fontsize=16)
    ax.set_xlabel("Predicted Labels", fontsize=14)
    ax.set_ylabel("True Labels", fontsize=14)
    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save and close the plot
    output_path = f"{task}/confusion_matrix_{mode}.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")
    
    

