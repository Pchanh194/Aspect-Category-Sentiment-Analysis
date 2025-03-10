import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from src.config.config import Config

class SyllableCharEmbedding(nn.Module):
    def __init__(self, syllable_vocab_size: int, char_vocab_size: int, embed_dim: int, hidden_dim: int):
        """Initialize syllable and character embedding module.
        
        Args:
            syllable_vocab_size: Size of syllable vocabulary
            char_vocab_size: Size of character vocabulary
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layer
        """
        super(SyllableCharEmbedding, self).__init__()
        self.syllable_embedding = nn.Embedding(syllable_vocab_size, embed_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, syllable_ids: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embedding module.
        
        Args:
            syllable_ids: Tensor of syllable indices
            char_ids: Tensor of character indices
            
        Returns:
            Tensor of combined embeddings
        """
        syllable_emb = self.syllable_embedding(syllable_ids)
        char_emb = self.char_embedding(char_ids)
        combined_emb = torch.cat((syllable_emb, char_emb), dim=-1)
        combined_emb = self.dropout(combined_emb)
        output, _ = self.lstm(combined_emb)
        return output

class ABSAModel(nn.Module):
    def __init__(self, syllable_vocab_size: int, char_vocab_size: int, num_labels: int, 
                 embed_dim: int, hidden_dim: int):
        """Initialize ABSA model with LoRA support.
        
        Args:
            syllable_vocab_size: Size of syllable vocabulary
            char_vocab_size: Size of character vocabulary
            num_labels: Number of output labels
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layer
        """
        super(ABSAModel, self).__init__()
        self.syllable_char_embed = SyllableCharEmbedding(
            syllable_vocab_size, char_vocab_size, embed_dim, hidden_dim // 2
        )
        
        # Load base model
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        
        # Configure and apply LoRA
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=["query", "value"],
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        # Apply LoRA to the model
        self.xlmr = get_peft_model(self.xlmr, lora_config)
        
        # Rest of the architecture
        self.lstm = nn.LSTM(
            hidden_dim + self.xlmr.config.hidden_size, 
            hidden_dim, 
            bidirectional=True, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, syllable_ids: torch.Tensor, char_ids: torch.Tensor, 
                input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ABSA model.
        
        Args:
            syllable_ids: Tensor of syllable indices
            char_ids: Tensor of character indices
            input_ids: Tensor of input token indices
            attention_mask: Tensor of attention mask
            
        Returns:
            Tensor of logits for each label
        """
        syllable_char_emb = self.syllable_char_embed(syllable_ids, char_ids)
        xlmr_output = self.xlmr(input_ids, attention_mask=attention_mask)[0]
        combined_emb = torch.cat((syllable_char_emb, xlmr_output), dim=-1)
        lstm_out, _ = self.lstm(combined_emb)
        lstm_out = self.dropout(lstm_out[:, 0, :])
        lstm_out = self.layer_norm(lstm_out)
        return self.fc(lstm_out)
        
    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}%"
        ) 