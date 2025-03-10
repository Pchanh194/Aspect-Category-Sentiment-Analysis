import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import wandb

class ModelVisualizer:
    @staticmethod
    def plot_roc_curve(labels, predictions, results):
        """Plot ROC curve for the model predictions."""
        fpr, tpr, _ = roc_curve(labels.ravel(), predictions.ravel())
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {results["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        wandb.log({"ROC Curve": wandb.Image(plt)})
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(labels, predictions, results, thresholds):
        """Plot Precision-Recall curve with threshold points."""
        plt.figure(figsize=(12, 8))
        
        # Plot threshold points
        for threshold in thresholds:
            if threshold != 'roc_auc' and threshold != 'average_precision':
                precision_micro = results[threshold]['precision_micro']
                recall_micro = results[threshold]['recall_micro']
                plt.scatter(recall_micro, precision_micro, 
                           label=f'Micro (t={threshold:.1f})', 
                           marker='o')
                
                precision_macro = results[threshold]['precision_macro']
                recall_macro = results[threshold]['recall_macro']
                plt.scatter(recall_macro, precision_macro, 
                           label=f'Macro (t={threshold:.1f})', 
                           marker='^')
        
        # Plot the curve
        precision, recall, _ = precision_recall_curve(labels.ravel(), predictions.ravel())
        plt.plot(recall, precision, color='blue', lw=2, 
                 label='Precision-Recall curve')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        wandb.log({"Precision-Recall Curve": wandb.Image(plt)})
        plt.close()

    @staticmethod
    def plot_model_weights(weights, biases):
        """Plot model weights and biases visualization."""
        plt.figure(figsize=(12, 6))

        # Plot weights heatmap
        plt.subplot(1, 2, 1)
        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar(label='Weight Value')
        plt.title('Weights of Final Layer')
        plt.xlabel('Output Neurons')
        plt.ylabel('Input Features')

        # Plot biases bar chart
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(biases)), biases, color='orange')
        plt.title('Biases of Final Layer')
        plt.xlabel('Output Neurons')
        plt.ylabel('Bias Value')

        plt.tight_layout()
        wandb.log({"Model Weights and Biases": wandb.Image(plt)})
        plt.close() 