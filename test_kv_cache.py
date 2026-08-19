"""Check cached decoding matches the original sliding-window forward pass."""

import torch

from language_model.model.model import Transformer


def main() -> None:
    model = Transformer(32, 4, 8, 2, 2).eval()
    token_ids = torch.tensor([1, 2, 3, 4, 5, 6])
    cache = None
    with torch.inference_mode():
        _, cache = model(token_ids[:2][None], use_cache=True)
        cached, _ = model(token_ids[2:][None], cache, use_cache=True)
        torch.testing.assert_close(cached, model(token_ids[None])[:, 2:])
        cache = None
        for index in range(len(token_ids)):
            inputs = token_ids[max(0, index - 3) : index + 1] if cache is None else token_ids[index : index + 1]
            cached, cache = model(inputs[None], cache, use_cache=True)
            expected = model(token_ids[max(0, index - 3) : index + 1][None])
            torch.testing.assert_close(cached[:, -1], expected[:, -1])
            if cache[0][0].shape[-2] == model.max_sequence_length:
                cache = None


if __name__ == "__main__":
    main()
