from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


VOCAB_SIZE = 16_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
