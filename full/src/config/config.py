import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # WandB Configuration
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    PROJECT_NAME = os.getenv('PROJECT_NAME', 'ABSA_model')
    RUN_NAME = os.getenv('RUN_NAME', 'default_run')

    # Model Configuration
    MAX_LEN = int(os.getenv('MAX_LEN', 128))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-5))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 50))
    WARMUP_RATIO = float(os.getenv('WARMUP_RATIO', 0.1))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.1))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', 8))
    EMBED_DIM = int(os.getenv('EMBED_DIM', 100))
    HIDDEN_DIM = int(os.getenv('HIDDEN_DIM', 200))

    # Training Configuration
    DEVICE = torch.device(os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    THRESHOLD = float(os.getenv('THRESHOLD', 0.7))
    T_0 = int(os.getenv('T_0', 10))
    T_MULT = int(os.getenv('T_MULT', 1))
    SAVE_STEPS = 100

    # Paths
    MODEL_CHECKPOINT_DIR = os.getenv('MODEL_CHECKPOINT_DIR', 'model_checkpoints')
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    BEST_MODEL_PATH = os.getenv('BEST_MODEL_PATH', 'best_absa_model.pth')

    # LoRA Configuration
    LORA_R = int(os.getenv('LORA_R', 16))
    LORA_ALPHA = int(os.getenv('LORA_ALPHA', 32))
    LORA_DROPOUT = float(os.getenv('LORA_DROPOUT', 0.1))

    @staticmethod
    def get_training_args():
        """Get training arguments for Hugging Face Trainer."""
        return {
            'output_dir': Config.MODEL_CHECKPOINT_DIR,
            'per_device_train_batch_size': Config.BATCH_SIZE,
            'per_device_eval_batch_size': Config.BATCH_SIZE,
            'gradient_accumulation_steps': Config.GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': Config.LEARNING_RATE,
            'max_grad_norm': 1.0,
            'optim': 'adamw_torch',
            'lr_scheduler_type': 'cosine',
            'weight_decay': Config.WEIGHT_DECAY,
            'warmup_ratio': Config.WARMUP_RATIO,
            'num_train_epochs': Config.NUM_EPOCHS,
            'logging_strategy': 'steps',
            'logging_steps': 10,
            'evaluation_strategy': 'steps',
            'eval_steps': 100,
            'save_strategy': 'steps',
            'save_steps': Config.SAVE_STEPS,
            'save_total_limit': None,
            'load_best_model_at_end': True,
            'report_to': 'wandb',
            'run_name': Config.RUN_NAME
        }