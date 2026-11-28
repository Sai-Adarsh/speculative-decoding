# Speculative decoding harness

Try speculative decoding with a CPU fixture or benchmark draft and target models with llama.cpp.

## Run locally

Run from the repo root with CMake 3.20+ and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/speculative-decoding
```

[Sampler details, llama.cpp setup, and benchmarks](docs/reference.md)
