from typing import Dict, List


class SimpleTokenizer:
    """A word-level tokenizer with special tokens."""

    def __init__(self) -> None:
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts by adding special tokens first."""
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        next_id = len(self.word_to_id)
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.word_to_id:
                    self.word_to_id[token] = next_id
                    next_id += 1

        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs using UNK for missing tokens."""
        unknown_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(token, unknown_id) for token in self._tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        """Convert list of token IDs back to text."""
        return " ".join(self.id_to_word.get(token_id, self.unk_token) for token_id in ids)


# Example usage / tests
tokenizer = SimpleTokenizer()

# build from sentences (list of strings)
tokenizer.build_vocab(["hello world", "I am here"])
# vocab now contains special tokens + words: hello, world, I, am, here

# encode a sentence (string) or list of tokens
ids = tokenizer.encode("I unknown")
# expected: [ tokenizer.word_to_id["I"], tokenizer.word_to_id.get("unknown", tokenizer.word_to_id[tokenizer.unk_token]) ]

# decode back
text = tokenizer.decode(ids)
# expected: "I <UNK>"  (unknown id decodes to "<UNK>")
print("ids", ids, tokenizer.unk_token, tokenizer.word_to_id[tokenizer.unk_token])

# assertions you can use (avoid hard-coding non-special ids)
assert tokenizer.word_to_id[tokenizer.pad_token] == 0
assert tokenizer.word_to_id[tokenizer.unk_token] == 1
assert ids[0] == tokenizer.word_to_id["I"]
assert ids[1] == tokenizer.word_to_id[tokenizer.unk_token]
assert text.split() == ["I", tokenizer.unk_token]
