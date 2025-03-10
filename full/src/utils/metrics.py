import numpy as np
from sklearn.metrics import (
    f1_score, classification_report, precision_score, recall_score,
    accuracy_score, roc_curve, auc, average_precision_score
)
import torch
from typing import Dict, Tuple, Any, List

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, thresholds: List[float]) -> Dict[str, Any]:
        """Calculate various metrics for model evaluation."""
        results = {}
        
        # Calculate metrics for each threshold
        for threshold in thresholds:
            binary_predictions = (predictions > threshold).astype(int)
            accuracy = np.mean([
                np.mean(y_true == y_pred) 
                for y_true, y_pred in zip(labels, binary_predictions)
            ])
            
            results[threshold] = {
                'accuracy': accuracy,
                'precision_micro': precision_score(labels, binary_predictions, average='micro', zero_division=1),
                'recall_micro': recall_score(labels, binary_predictions, average='micro', zero_division=1),
                'f1_micro': f1_score(labels, binary_predictions, average='micro', zero_division=1),
                'precision_macro': precision_score(labels, binary_predictions, average='macro', zero_division=1),
                'recall_macro': recall_score(labels, binary_predictions, average='macro', zero_division=1),
                'f1_macro': f1_score(labels, binary_predictions, average='macro', zero_division=1)
            }
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(labels.ravel(), predictions.ravel())
        results['roc_auc'] = auc(fpr, tpr)
        
        # Calculate average precision
        results['average_precision'] = average_precision_score(labels, predictions, average='micro')
        
        return results

    @staticmethod
    def calculate_class_weights(labels: List[str], aspect_sentiment_to_idx: Dict[str, int]) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        from sklearn.preprocessing import MultiLabelBinarizer
        
        mlb = MultiLabelBinarizer()
        binary_labels = mlb.fit_transform([label.split('\t') for label in labels])
        
        n_samples = len(binary_labels)
        weights = []
        
        for i in range(len(aspect_sentiment_to_idx)):
            n_pos = np.sum(binary_labels[:, i])
            n_neg = n_samples - n_pos
            weight = n_samples / (2 * n_pos) if n_pos > 0 else 0.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)

    @staticmethod
    def print_classification_report(predictions: np.ndarray, labels: np.ndarray, 
                                  idx_to_aspect_sentiment: Dict[int, str], threshold: float = 0.7):
        """Print detailed classification report."""
        binary_predictions = (predictions > threshold).astype(int)
        label_names = [idx_to_aspect_sentiment[i] for i in range(len(idx_to_aspect_sentiment))]
        
        print("\nClassification Report:")
        print(classification_report(labels, binary_predictions, 
                                  target_names=label_names, zero_division=1))

    @staticmethod
    def get_test_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.7) -> Dict[str, float]:
        """Calculate metrics for test set."""
        binary_predictions = (predictions > threshold).astype(int)
        
        return {
            'accuracy': np.mean([
                np.mean(true == pred) 
                for true, pred in zip(labels, binary_predictions)
            ]),
            'precision': precision_score(labels, binary_predictions, average='micro'),
            'recall': recall_score(labels, binary_predictions, average='micro'),
            'f1': f1_score(labels, binary_predictions, average='micro')
        }