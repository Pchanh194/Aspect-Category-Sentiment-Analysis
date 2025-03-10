# Aspect-based Sentiment Analysis for Vietnamese

This project implements an advanced Aspect-based Sentiment Analysis (ABSA) system for Vietnamese text using a hybrid architecture that combines XLM-RoBERTa with syllable and character-level embeddings, enhanced with LoRA (Low-Rank Adaptation) fine-tuning.

## Features

- Hybrid model architecture combining:
  - XLM-RoBERTa for contextual embeddings
  - Syllable-level embeddings for Vietnamese word structure
  - Character-level embeddings for fine-grained features
  - LoRA adaptation for efficient fine-tuning
- Multi-label classification for aspect-sentiment pairs
- Mixed precision training for improved performance
- Comprehensive evaluation metrics and visualization
- Wandb integration for experiment tracking


## Project Structure

```
.
├── src/
│   ├── config/
│   │   └── config.py           # Configuration management
│   ├── data/
│   │   ├── dataset.py          # Dataset and data loading
│   │   └── data_generator.py   # Data generation utilities
│   ├── models/
│   │   └── model.py           # Model architecture
│   ├── tokenizers/
│   │   └── tokenizers.py      # Custom tokenizers
│   └── utils/
│       ├── early_stopping.py  # Early stopping utility
│       ├── metrics.py         # Metrics calculation
│       ├── trainer.py         # Training utilities
│       └── visualization.py   # Visualization tools
├── data/                      # Data directory
│   ├── train.jsonl           # Training data
│   ├── val.jsonl             # Validation data
│   └── test.jsonl            # Test data
├── model_checkpoints/         # Model checkpoints
├── train.py                   # Main training script
├── requirements.txt           # Project dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

## Data Format and Organization

The project uses multiple data sources organized in different directories:

1. **preprocess_data/**: Contains the preprocessed data ready for training
   - Includes train, validation, and test splits in JSONL format
   - Data has been cleaned and formatted according to project requirements

2. **raw_generated/**: Contains raw generated data
   - Includes data generated using various methods
   - Requires preprocessing before use

3. **acsa_msNgan/**: ACSA dataset from msNgan
   - Original ACSA dataset with aspect-based annotations
   - Used as one of the main data sources

4. **vlsp2018/**: VLSP 2018 dataset
   - Vietnamese Language and Speech Processing dataset
   - Contains sentiment analysis annotations

Each data file should follow the JSONL format with this structure:
```json
{
    "text": "Máy chạy rất mượt, pin trâu, camera chụp đẹp",
    "labels": ["PERFORMANCE#POSITIVE", "BATTERY#POSITIVE", "CAMERA#POSITIVE"]
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd acsa-vietnamese
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data model_checkpoints
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Data Format

The data should be in JSONL format with the following structure:
```json
{
    "text": "Máy chạy rất mượt, pin trâu, camera chụp đẹp",
    "labels": ["PERFORMANCE#POSITIVE", "BATTERY#POSITIVE", "CAMERA#POSITIVE"]
}
```

Available aspect-sentiment labels:
- BATTERY#[NEGATIVE|NEUTRAL|POSITIVE]
- CAMERA#[NEGATIVE|NEUTRAL|POSITIVE]
- DESIGN#[NEGATIVE|NEUTRAL|POSITIVE]
- FEATURES#[NEGATIVE|NEUTRAL|POSITIVE]
- GENERAL#[NEGATIVE|NEUTRAL|POSITIVE]
- PERFORMANCE#[NEGATIVE|NEUTRAL|POSITIVE]
- PRICE#[NEGATIVE|NEUTRAL|POSITIVE]
- SCREEN#[NEGATIVE|NEUTRAL|POSITIVE]
- SER&ACC#[NEGATIVE|NEUTRAL|POSITIVE]
- STORAGE#[NEGATIVE|NEUTRAL|POSITIVE]

## Model Architecture

The model combines three main components:

1. **XLM-RoBERTa Base**:
   - Pre-trained multilingual transformer
   - Fine-tuned with LoRA for efficiency
   - Handles contextual understanding

2. **Syllable-Character Embeddings**:
   - Syllable-level embeddings for Vietnamese words
   - Character-level embeddings for subword features
   - BiLSTM for sequence processing

3. **Classification Head**:
   - BiLSTM for feature combination
   - Layer normalization
   - Linear layer for multi-label classification

## Training

1. Prepare your data in the required JSONL format

2. Configure training parameters in `.env`:
```env
WANDB_API_KEY=your_wandb_key
MAX_LEN=128
BATCH_SIZE=64
LEARNING_RATE=1e-5
NUM_EPOCHS=50
```

3. Run training:
```bash
python train.py
```

## Configuration

Key configuration parameters in `.env`:

```env
# Model Configuration
MAX_LEN=128
BATCH_SIZE=64
LEARNING_RATE=1e-5
NUM_EPOCHS=50
EMBED_DIM=100
HIDDEN_DIM=200

# Training Configuration
THRESHOLD=0.7
T_0=10
T_MULT=1

# LoRA Configuration
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
```

## Evaluation Metrics

The model is evaluated using:
- Micro and Macro F1 scores
- Precision and Recall
- ROC-AUC
- Average Precision
- Classification Report per aspect-sentiment

## Visualization

The training process includes:
- ROC curves
- Precision-Recall curves
- Model weights visualization
- Training metrics tracking via Wandb

## Results

Example classification report:
```
              precision    recall  f1-score   support
BATTERY#POS      0.85      0.82      0.83       100
CAMERA#POS       0.88      0.85      0.86       150
...
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- XLM-RoBERTa: [Hugging Face](https://huggingface.co/xlm-roberta-large)
- LoRA: [Microsoft](https://github.com/microsoft/LoRA)
- PEFT: [Hugging Face](https://github.com/huggingface/peft)

## Contact

For questions and feedback, please open an issue or contact the maintainers.