# Speculative decoding harness

A draft model proposes a window of tokens, and the target verifies them in one call. The C++ sampler accepts each proposal with probability `min(1, p(token)/q(token))`, where `p` is the target distribution and `q` is the draft distribution. On rejection it samples from the normalized positive part of `p - q`. If every proposal passes, it draws one more token from the target. The accepted prefix and the new token become the next context. [Leviathan et al.](https://arxiv.org/abs/2211.17192)

## CPU fixture

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/speculative-decoding
./build/speculative-decoding 500 256 8
```

Two four-token models with different conditional distributions exercise the sampler. The CSV compares target verification calls, proposal acceptance, and sampled token frequencies. The baseline and speculative runs use independent random streams, so individual sequences can differ. The fixture's target evaluates the proposed columns on the CPU. Call counts measure batching behavior.

## 2B draft and 70B target with llama.cpp

The optional C++ HTTP client needs libcurl development files. Use a CUDA-enabled `llama-server` whose help includes `--spec-draft-sampling`. Set `TARGET_GGUF` to a 70B checkpoint and `DRAFT_GGUF` to a roughly 2B checkpoint with the same vocabulary and token ID mapping. Choose quantization and GPU layers to fit your hardware. [llama-server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_LLAMA_SERVER=ON
cmake --build build

# Start this in a separate terminal for the baseline.
llama-server -m "$TARGET_GGUF" --host 127.0.0.1 --port 8080 -ngl all --spec-type none
```

Capture the baseline when the server is ready:

```sh
./build/speculative-decoding --server http://127.0.0.1:8080 'Explain continuous batching.' 256 > baseline.json
```

Stop the baseline server to free the target's memory, then start the speculative server:

```sh
# Start this in a separate terminal for the speculative run.
llama-server -m "$TARGET_GGUF" -md "$DRAFT_GGUF" --host 127.0.0.1 --port 8080 \
  -ngl all -ngld all --spec-type draft-simple --spec-draft-n-max 4 --spec-draft-sampling probabilistic
```

```sh
./build/speculative-decoding --server http://127.0.0.1:8080 'Explain continuous batching.' 256 > speculative.json
jq '.timings' baseline.json speculative.json
```

The server drafts and verifies tokens. The client saves completion timings and writes end-to-end latency to stderr. Warm both configurations and repeat over several prompts. Record accepted proposals, generation tokens per second, prompt characteristics, and VRAM to assess speedup on your hardware.
