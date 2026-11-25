# PrivCirNet · Block-Circulant Private Inference

Run the two-party block-circulant linear and convolution operators from PrivCirNet (NeurIPS 2024) on generated CPU inputs. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ca9873918aa72e9033041f76e77b5c15-Abstract-Conference.html) · [Linear layer](../../Layer/LinearLayer/include/LinearLayer/CirLinear.h) · [Convolution](../../Layer/LinearLayer/src/CirConv.cpp)

## Contents

1. [Build the checks](#build-the-checks)
2. [Run both parties](#run-both-parties)
3. [Benchmarks and pretrained models](#benchmarks-and-pretrained-models)

## Build the checks

From the repository root, install the [CPU dependencies](../../Extern/README.md), then build both checks.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target test_cir_linear test_cir_conv -j8
```

## Run both parties

For each row, start the server in one terminal, then run the client in a second terminal on the same host. The server compares reconstructed outputs with a plaintext reference and prints a result for each case.

| Operator | Terminal 1 · server | Terminal 2 · client |
|---|---|---|
| Block-circulant linear layer | `./build/Test/test_cir_linear r=1 p=1234` | `./build/Test/test_cir_linear r=2 p=1234` |
| Block-circulant convolution | `./build/Test/test_cir_conv r=1 p=1235` | `./build/Test/test_cir_conv r=2 p=1235` |

Each server should print seven `PASS` rows.

## Benchmarks and pretrained models

For rotation counts and timings, use the [linear-layer benchmark](../Benchmarks/README.md). For training, layer-wise block-size selection and pretrained models, see the [official PrivCirNet repository](https://github.com/Tianshi-Xu/PrivCirNet).
