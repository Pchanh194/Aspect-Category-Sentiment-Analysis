import torch
import torch.cuda.amp as amp
from tqdm import tqdm
import os
import wandb
from typing import Dict, Any, Optional, Tuple
from src.config.config import Config
from src.utils.metrics import MetricsCalculator

class ModelTrainer:
    def __init__(self, model, optimizer, criterion, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.scaler = amp.GradScaler()
        self.metrics_calculator = MetricsCalculator()

    def save_model(self, path: str, epoch: int, val_loss: float, f1: float, 
                  syllable_vocab_size: int, char_vocab_size: int, 
                  num_labels: int, aspect_sentiment_to_idx: Dict[str, int]):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            'f1': f1,
            'syllable_vocab_size': syllable_vocab_size,
            'char_vocab_size': char_vocab_size,
            'num_labels': num_labels,
            'aspect_sentiment_to_idx': aspect_sentiment_to_idx
        }, path)

    def save_lora_model(self, path: str):
        """Save LoRA model and adapter weights."""
        # Save the LoRA adapter weights separately
        self.model.xlmr.save_pretrained(path)
        
        # Save the full model state
        full_state_dict = {
            'syllable_char_embed': self.model.syllable_char_embed.state_dict(),
            'lstm': self.model.lstm.state_dict(),
            'layer_norm': self.model.layer_norm.state_dict(),
            'fc': self.model.fc.state_dict(),
        }
        torch.save(full_state_dict, f"{path}/full_model_state.bin")

    def train_epoch(self, train_loader, global_step: int) -> Tuple[float, int]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Move data to device
            syllable_ids = batch['syllable_ids'].to(self.device)
            char_ids = batch['char_ids'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Use mixed precision training
            with amp.autocast():
                outputs = self.model(syllable_ids, char_ids, input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            global_step += 1

            if global_step % Config.SAVE_STEPS == 0:
                save_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, f'model_step_{global_step}')
                self.save_lora_model(save_path)
                print(f"Model saved at step {global_step}")

        self.scheduler.step()
        return total_loss / len(train_loader), global_step

    def evaluate(self, data_loader, thresholds) -> Tuple[float, Dict[str, Any], torch.Tensor, torch.Tensor]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                syllable_ids = batch['syllable_ids'].to(self.device)
                char_ids = batch['char_ids'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(syllable_ids, char_ids, input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        results = self.metrics_calculator.calculate_metrics(all_predictions, all_labels, thresholds)
        
        return total_loss / len(data_loader), results, all_predictions, all_labels

    def log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict[str, Any], 
                   threshold: float = 0.7):
        """Log metrics to wandb."""
        metrics = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Validation Loss": val_metrics['loss'],
            "Val Accuracy": val_metrics[threshold]['accuracy'],
            "Val Micro Precision": val_metrics[threshold]['precision_micro'],
            "Val Micro Recall": val_metrics[threshold]['recall_micro'],
            "Val Micro F1": val_metrics[threshold]['f1_micro'],
            "Val Macro Precision": val_metrics[threshold]['precision_macro'],
            "Val Macro Recall": val_metrics[threshold]['recall_macro'],
            "Val Macro F1": val_metrics[threshold]['f1_macro'],
            "Learning Rate": self.optimizer.param_groups[0]['lr'],
            "Val AUC-ROC": val_metrics['roc_auc'],
            "Val Average Precision": val_metrics['average_precision']
        }
        wandb.log(metrics)

    def inference_samples(self, data_loader, idx_to_aspect_sentiment: Dict[int, str], 
                        num_samples: int = 5) -> list:
        """Run inference on sample data."""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(results) >= num_samples:
                    break
                    
                syllable_ids = batch['syllable_ids'].to(self.device)
                char_ids = batch['char_ids'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(syllable_ids, char_ids, input_ids, attention_mask)
                predictions = (torch.sigmoid(outputs) > Config.THRESHOLD).int()
                
                for i, (text, pred) in enumerate(zip(batch['text'], predictions)):
                    if len(results) >= num_samples:
                        break
                        
                    pred_aspects = [idx_to_aspect_sentiment[j] for j, p in enumerate(pred) if p == 1]
                    results.append({
                        'text': text,
                        'predicted_aspects': pred_aspects
                    })
        
        return results 