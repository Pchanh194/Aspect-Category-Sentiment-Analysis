from collections import Counter
from typing import List, Dict, Set

class SimpleTokenizer:
    def __init__(self):
        """Initialize base tokenizer with special tokens."""
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>'}

    def fit(self, texts: List[str]):
        """Fit tokenizer on a list of texts.
        
        Args:
            texts: List of text strings to fit on
        """
        tokens = [token for text in texts for token in self.tokenize(text)]
        token_counts = Counter(tokens)
        
        # Add tokens to vocabulary, skipping tokens already in vocabulary
        for token in token_counts:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of tokens.
        
        Args:
            text: Text string to tokenize
            
        Returns:
            List of tokens
        """
        raise NotImplementedError("Subclasses must implement tokenize method")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to list of token ids.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token ids
        """
        return [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of token ids to list of tokens.
        
        Args:
            ids: List of token ids
            
        Returns:
            List of token strings
        """
        return [self.id_to_token.get(idx, '<UNK>') for idx in ids]

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary.
        
        Returns:
            Dictionary mapping tokens to ids
        """
        return self.token_to_id.copy()

    def get_vocab_size(self) -> int:
        """Get size of vocabulary.
        
        Returns:
            Number of tokens in vocabulary
        """
        return len(self.token_to_id)

    def __len__(self) -> int:
        """Get size of vocabulary.
        
        Returns:
            Number of tokens in vocabulary
        """
        return self.get_vocab_size()

class SyllableTokenizer(SimpleTokenizer):
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into syllables (words for Vietnamese).
        
        Args:
            text: Text string to tokenize
            
        Returns:
            List of syllable tokens
        """
        return text.lower().split()

class CharacterTokenizer(SimpleTokenizer):
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters.
        
        Args:
            text: Text string to tokenize
            
        Returns:
            List of character tokens
        """
        return list(text.lower())