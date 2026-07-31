from websem_client import WordPieceTokenizer, normalize, pre_tokenize


def test_bert_normalization_and_pre_tokenization() -> None:
    assert normalize("Café\x00\x07\t中！") == "cafe  中 ！"
    assert pre_tokenize(normalize("Naïve can't + 中")) == [
        "naive",
        "can",
        "'",
        "t",
        "+",
        "中",
    ]


def test_wordpiece_is_greedy_and_drops_unknown_without_special_tokens() -> None:
    tokenizer = WordPieceTokenizer(["[UNK]", "play", "##ing", "player", "'", "t", "中"])

    assert tokenizer.tokenize_word("playing") == ["play", "##ing"]
    assert tokenizer.encode("Playing mystery 中") == [1, 2, 6]
    assert tokenizer.encode("mystery") == []
    assert tokenizer.encode("x" * 101) == []
