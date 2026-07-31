from unicodedata import category
from unicodedata import normalize as unicode_normalize

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)


def _is_cjk(character: str) -> bool:
    code_point = ord(character)
    return any(start <= code_point <= end for start, end in _CJK_RANGES)


def normalize(text: str) -> str:
    cleaned: list[str] = []
    for character in text:
        code_point = ord(character)
        if code_point in (0, 0xFFFD):
            continue
        if character in "\t\n\r":
            cleaned.append(" ")
            continue
        if category(character).startswith("C"):
            continue
        if _is_cjk(character):
            cleaned.extend((" ", character, " "))
            continue
        cleaned.append(character)

    decomposed = unicode_normalize("NFD", "".join(cleaned).lower())
    return "".join(character for character in decomposed if category(character) != "Mn")


def pre_tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for character in text:
        character_category = category(character)
        if character.isspace():
            if current:
                tokens.append("".join(current))
                current.clear()
            continue
        if character_category.startswith(("P", "S")):
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(character)
            continue
        current.append(character)
    if current:
        tokens.append("".join(current))
    return tokens


class WordPieceTokenizer:
    def __init__(
        self,
        vocabulary: list[str] | tuple[str, ...],
        *,
        unknown_token: str = "[UNK]",
        continuation_prefix: str = "##",
        max_chars_per_word: int = 100,
    ) -> None:
        if max_chars_per_word < 1:
            raise ValueError("max_chars_per_word must be positive")
        self.vocabulary = tuple(vocabulary)
        self.unknown_token = unknown_token
        self.continuation_prefix = continuation_prefix
        self.max_chars_per_word = max_chars_per_word
        self._vocabulary_index = {token: token_id for token_id, token in enumerate(self.vocabulary)}
        self._unknown_id = self._vocabulary_index.get(unknown_token, -1)

    def tokenize_word(self, word: str) -> list[str]:
        if len(word) > self.max_chars_per_word:
            return [self.unknown_token]

        sub_tokens: list[str] = []
        start = 0
        while start < len(word):
            matched: str | None = None
            end = len(word)
            while end > start:
                candidate = word[start:end]
                if start > 0:
                    candidate = f"{self.continuation_prefix}{candidate}"
                if candidate in self._vocabulary_index:
                    matched = candidate
                    break
                end -= 1
            if matched is None:
                return [self.unknown_token]
            sub_tokens.append(matched)
            start = end
        return sub_tokens

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for word in pre_tokenize(normalize(text)):
            for token in self.tokenize_word(word):
                token_id = self._vocabulary_index.get(token)
                if token_id is not None and token_id != self._unknown_id:
                    token_ids.append(token_id)
        return token_ids
