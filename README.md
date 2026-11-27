# Speculative decoding harness

A draft model proposes a window of tokens. The target needs one next-token distribution per proposal and one bonus distribution. A transformer can produce those distributions in one forward pass over the window. The C++ sampler accepts each proposal with probability `min(1, p(token)/q(token))`, where `p` is the target distribution and `q` is the draft distribution. On rejection it samples from the normalized positive part of `p - q`. If every proposal passes, it draws one more token from the target. The accepted prefix and the new token become the next context. [Leviathan et al.](https://arxiv.org/abs/2211.17192)

## CPU fixture

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/speculative-decoding
./build/speculative-decoding 500 256 8
```

Two four-token models exercise the sampler. The target depends on the previous two tokens. The fixture evaluates each target distribution serially and reports logical target requests and actual target positions separately. Wall time covers sequence generation and excludes histogram collection. The two samplers alternate execution order and use independent seeded random streams.

For the [recorded run](evidence/fixture-5000-128-4.csv), each sampler emitted 640,000 tokens. Speculation reduced logical target requests from 640,000 to 164,731 while increasing serial target positions to 810,775. The fixture took 32.04 ms autoregressively and 68.47 ms with speculation.

The evaluator enumerates the exact three-token prefix probabilities under the target model and compares them to one prefix per trial. It also reports weighted total variation for the empirical next-token probabilities conditioned on the previous two tokens. The recorded joint prefix error was 0.03231 for the baseline and 0.03223 for speculation. These are finite-sample checks against a known distribution.

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


## Measured local transformer pair

The [seven recorded requests per mode](evidence/tinystories-loopback.csv) used llama.cpp `b12810` / `ca123ba76` on an Apple M3 Pro. The target was the F32 `stories15M.gguf` TinyStories checkpoint (24.4M total parameters as reported by llama.cpp). The draft was the same checkpoint calibrated to Q4_K_M, with the matching tokenizer. The [calibration project](https://github.com/Sai-Adarsh/fp4-calibration) records its input hashes and corpus recipe.

Both configurations used one slot, four threads, a 128-token context, and zero offloaded model layers. The speculative run used a four-token maximum draft and probabilistic verification. After one warmup, each configuration completed the same 11-token prompt and generated 64 tokens with EOS suppression.

| Mode | Server decode tok/s, median | Client completion ms, median | Draft accepted |
|---|---:|---:|---:|
| Autoregressive | 1,180.86 | 58.64 | — |
| Speculative | 1,038.85 | 64.39 | 350 / 350 |

The draft passed every proposal in this run. The extra work still cost about 12% of decode throughput on this model pair. That result covers one prompt, one fixed seed, and this local setup. The 2B/70B CUDA run remains unmeasured.

Reproduce the local shape with the server commands above, changing target flags to `-ngl 0 -c 128 -t 4 -tb 4 -np 1` and draft flags to `-ngld 0 -td 4 -tbd 4`. For each configuration, discard one warmup, then repeat:

```sh
./build/speculative-decoding --server http://127.0.0.1:8080 \
  'Once upon a time, there was a little dog' 64 --ignore-eos
```

The client emits full server JSON on stdout and end-to-end elapsed milliseconds on stderr. Compare `timings.predicted_per_second`, `timings.draft_n`, and `timings.draft_n_accepted`. The two measured files had these SHA-256 hashes:

- F32 target: `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`
- Q4_K_M draft: `38bf32ea3054a7badf42b0374b61442cfbf2c8c3d4f2d89568a903bd863e4828`

## Tests and coverage

```sh
cmake -S . -B build -DENABLE_LLAMA_SERVER=ON -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The fixture test checks token accounting, accepted and rejected proposals, and joint error across emitted sequences. The HTTP test uses a local fake server to check request escaping, EOS suppression, timing response passthrough, and failure handling for HTTP errors, empty replies, and oversized responses. Neither test downloads models or contacts an external service.

For merged line and branch coverage on macOS, exercise both build configurations:

```sh
cov_dir=$(mktemp -d)
mkdir "$cov_dir/raw"
for enabled in ON OFF; do
  cmake -S . -B "$cov_dir/$enabled" -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_LLAMA_SERVER="$enabled" \
    -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate"
  cmake --build "$cov_dir/$enabled"
  LLVM_PROFILE_FILE="$cov_dir/raw/%m-%p.profraw" ctest --test-dir "$cov_dir/$enabled" --output-on-failure
done
xcrun llvm-profdata merge -sparse "$cov_dir"/raw/*.profraw -o "$cov_dir/merged.profdata"
xcrun llvm-cov report "$cov_dir/ON/speculative-decoding" \
  -object "$cov_dir/OFF/speculative-decoding" -instr-profile "$cov_dir/merged.profdata"
```

The two builds covered 306 of 310 production lines (98.71%) and 126 of 132 branches (95.45%). The uncovered lines handle libcurl initialization and response-buffer allocation failures.
