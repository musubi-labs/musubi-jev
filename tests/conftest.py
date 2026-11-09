"""Fixtures: a tiny randomly initialised checkpoint built locally (no download), so the loader, Kev and the server run in CI.
The tokenizer is a byte-level BPE trained on a few lines, with the five Kev delimiters and Qwen-style added tokens (`<|im_start|>`,
`<think>`, `<tool_call>`, ...); the backbone is a 2-layer Qwen2 with 32 hidden units; the adapter is a rank-4 LoRA with random weights."""
import pytest
import torch

ADDED_SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
ADDED_PLAIN = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>"]
CORPUS = ["the quick brown fox jumps over the lazy dog", "I was charged twice. Please fix this ASAP.", "Is this ticket about billing? yes no",
          "returns shipping billing calm frustrated angry", "Order 4411 arrived late and the box was crushed."] * 4


def build_tokenizer():
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from kev.model import SPECIAL
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tk.add_special_tokens(SPECIAL + ADDED_SPECIAL)
    tk.add_tokens([AddedToken(t, special=False) for t in ADDED_PLAIN])
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<|endoftext|>", eos_token="<|endoftext|>")


@pytest.fixture(scope="session")
def tiny_tok():
    return build_tokenizer()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Path of a checkpoint in the layout kev.train writes: adapter files and head.pt (temperature 2.0), base in a sibling directory."""
    from transformers import AutoModelForCausalLM, Qwen2Config
    from kev.model import DecisionModel
    root = tmp_path_factory.mktemp("tiny-kev")
    torch.manual_seed(0)
    tok = build_tokenizer()
    base, run = root / "base", root / "run"
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=32768, tie_word_embeddings=True)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(base); tok.save_pretrained(base)
    m = DecisionModel(str(base), tok, "cpu", lora=4, head_dim=8)
    with torch.no_grad():
        for n, p in m.lm.named_parameters():
            if "lora_B" in n: p.copy_(torch.randn_like(p) * 0.05)
    m.lm.save_pretrained(run); tok.save_pretrained(run)
    torch.save({"head": m.head.state_dict(), "base": str(base), "base_revision": None, "lora": 4, "head_dim": 8, "option_isolation": False,
                "weights_dtype": None, "temperature": 2.0}, run / "head.pt")
    return str(run)


@pytest.fixture(scope="session")
def tiny_kev(tiny_checkpoint):
    from kev.config import Config
    from kev.runtime import Kev
    return Kev.from_pretrained(tiny_checkpoint, device="cpu", config=Config(prefix_min_tokens=0))
