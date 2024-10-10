import numpy as np
import time
from tqdm import tqdm
from gpt2.utils import load_encoder_hparams_and_params
from gpt2.gpt import gpt2, softmax
from helper import get_sample, max_fn
import functools


def auto_reg_sampling(input_seq, model, N_future):
    n = len(input_seq)
    T = len(input_seq) + N_future

    with tqdm(total=N_future, desc="autoreg sampling") as pbar:
        while n < T:
            input_seq = np.append(input_seq, get_sample(model(input_seq)[-1]))
            n += 1
            pbar.update(1)
    return input_seq


def spec_sampling(input_seq, draft_model, target_model, N_future, k):
    n = len(input_seq)
    T = len(input_seq) + N_future

    with tqdm(total=N_future, desc="Spec Sampling") as pbar:
        while n < T:
            prev_n = n
            # step1: autoreg generate from draft model and sample p
            input_draft = input_seq
            for _ in range(k):
                p = draft_model(input_draft)  # out logits
                input_draft = np.append(input_draft, get_sample(p[-1]))

            # step2: input the whole seq of draft to target model
            q = target_model(input_draft)

            # step3: Acceptance/ Rejection based on the p/q ratio
            all_generated_tokens_accepted = True
            for _ in range(k):
                i = n - 1
                j = input_draft[i + 1]

                if np.random.random() < min(1, q[i][j] / p[i][j]):  # accepted
                    input_seq = np.append(input_seq, j)
                    n += 1
                else:  # rejected ---> resample from q-p
                    input_seq = np.append(input_seq, get_sample(max_fn(q[i] - p[i])))
                    n += 1
                    all_generated_tokens_accepted = False
                    break

            # step 4
            if all_generated_tokens_accepted:
                input_seq = np.append(input_seq, get_sample(q[-1]))
                n += 1

            # for the bar
            pbar.update(n - prev_n)
            assert n == len(input_seq), f"{n} {len(input_seq)}"
    return input_seq


def create_model_fn(params, hparams, temperature, eps=1e-10):
    f = functools.partial(gpt2, **params, n_head=hparams["n_head"])

    def model_fn(inputs):
        logits = f(inputs)
        logits = logits / (temperature + eps)  # eps to avoid division by zero
        probs = softmax(logits)
        return probs

    return model_fn


def main(
    prompt: str = "Quantization also improves latency and throughput but suffer from perf",
    n_tokens_to_generate: int = 40,
    draft_model_size: str = "124M",
    target_model_size: str = "355M",
    models_dir: str = "models",
    K: int = 4,
    temperature: float = 0.0,
    seed: int = 123,
):
    # seed numpy rng
    np.random.seed(seed)

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, draft_hparams, draft_params = load_encoder_hparams_and_params(
        draft_model_size, models_dir
    )
    _, target_hparams, target_params = load_encoder_hparams_and_params(
        target_model_size, models_dir
    )
    draft_model = create_model_fn(draft_params, draft_hparams, temperature)
    target_model = create_model_fn(target_params, target_hparams, temperature)

    # encode inputs
    input_ids = encoder.encode(prompt)

    def run_sampling_fn(decode_fn, input_seq, **kwargs):
        start = time.perf_counter()
        output_ids = decode_fn(input_seq=input_seq, **kwargs)
        text = encoder.decode(output_ids)
        elapsed_time = time.perf_counter() - start
        return text, elapsed_time

    # autoregressive sampling
    autoregressive_text, autoregressive_time = run_sampling_fn(
        auto_reg_sampling,
        input_seq=input_ids,  # Pass correct parameter
        model=target_model,
        N_future=n_tokens_to_generate,  # Use N_future instead of N
    )

    # speculative sampling
    speculative_text, speculative_time = run_sampling_fn(
        spec_sampling,
        input_seq=input_ids,  # Pass correct parameter
        target_model=target_model,
        draft_model=draft_model,
        N_future=n_tokens_to_generate,  # Use N_future instead of N
        k=K,
    )

    # print results
    print()
    print("Autoregressive Decoding:")
    print("-" * 50)
    print(f"Time = {autoregressive_time:.2f}s")
    print(f"Text = {autoregressive_text}")
    print()
    print("Speculative Decoding:")
    print("-" * 50)
    print(f"Time = {speculative_time:.2f}s")
    print(f"Text = {speculative_text}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
