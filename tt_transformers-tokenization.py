import numpy as np
from typing import List, Dict

from traitlets import default

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {
          self.pad_token : 0,
          self.unk_token : 1,
          self.bos_token : 2,
          self.eos_token : 3,
        }
        texts = " ".join(texts)
        texts = texts.split()
        vocab = set(texts)
        i = 4
        for w in vocab:
            self.word_to_id[w] = i
            i+=1
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}
        self.vocab_size = len(self.id_to_word) - 4
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE

        text = text.split()
        default = self.word_to_id[self.unk_token]
        ret =[]
        for w in text:
            ret.append(self.word_to_id.get(w,default))
        return ret

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        ret =[]
        default = self.encode(self.unk_token)
        for i in ids:
            ret.append(self.id_to_word.get(i,default))
        ret = " ".join(ret)
        return ret

# Example usage / tests
tokenizer = SimpleTokenizer()

# build from sentences (list of strings)
tokenizer.build_vocab(["hello world", "I am here"])
# vocab now contains special tokens + words: hello, world, I, am, here

# encode a sentence (string) or list of tokens
ids = tokenizer.encode("I unknown")            # or tokenizer.encode(["I","unknown"])
# expected: [ tokenizer.word_to_id["I"], tokenizer.word_to_id.get("unknown", tokenizer.word_to_id[tokenizer.unk_token]) ]

# decode back
text = tokenizer.decode(ids)
# expected: "I <UNK>"  (unknown id decodes to "<UNK>")
print("ids",ids, tokenizer.unk_token, tokenizer.word_to_id[tokenizer.unk_token] )
# assertions you can use (avoid hard-coding non-special ids)
assert tokenizer.word_to_id[tokenizer.pad_token] == 0
assert tokenizer.word_to_id[tokenizer.unk_token] == 1
assert ids[0] == tokenizer.word_to_id["I"]
assert ids[1] == tokenizer.word_to_id[tokenizer.unk_token]
assert text.split() == ["I", tokenizer.unk_token]