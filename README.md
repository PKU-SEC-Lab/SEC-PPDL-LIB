![SEC-PPDL by PKU SEC Lab: homomorphic convolution followed by an MPC activation](docs/assets/sec-ppdl.jpg)

# SEC-PPDL

[![Version: development](https://img.shields.io/badge/version-development-176B70)](https://github.com/PKU-SEC-Lab/SEC-PPDL-LIB/tags)
[![License: unspecified](https://img.shields.io/badge/license-unspecified-626D76)](#license)

## About

SEC-PPDL is the PKU SEC Lab C++ library for two-party private neural network inference. Combine homomorphic encryption (HE) for linear layers with secure multiparty computation (MPC) for activations and comparisons. Applications can link individual operators or use the shared protocol runtime.

[Quick start](#quick-start) · [Run PrivCirNet](#run-privcirnet) · [Operator usage](#usage) · [Experiments](#performance) · [Publications](#publications) · [Contributor guide](CONTRIBUTING.md)

## News

- OptiPrime (MICRO 2026) adds OptiEncode convolution on CPU and CUDA. The repository also includes a subsequent packed depthwise extension for MobileNetV2. [Project page](https://jerryfishy.github.io/OptiPrime/) · [Depthwise layout](Experiments/OptiPrime/depthwise.md)

## Current support

| Component | Available in the library | Entry point |
|---|---|---|
| Encrypted convolution | OptiEncode dense and multiplier-one depthwise convolution on CPU and CUDA; Cheetah and block-circulant convolution | [OptiEncode](Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h) · [Other convolutions](Layer/LinearLayer/include/LinearLayer/Conv.h) |
| Linear algebra | Packed matrix multiplication and block-circulant linear layers | [Linear](Layer/LinearLayer/include/LinearLayer/Linear.h) · [CirLinear](Layer/LinearLayer/include/LinearLayer/CirLinear.h) |
| Secure nonlinear operators | ReLU, pooling, fixed-point comparison, selection and truncation | [Layers](Layer/NonlinearLayer/include/NonlinearLayer) · [Fixed-point protocols](Operator/NonlinearOperator/include/NonlinearOperator/FixPoint.h) |
| Two-party runtime | HE key setup, IKNP or Ferret OT, communication counters | [Runtime](Runtime/include/Runtime/Primitive.h) |

## Quick start

From a fresh clone, follow the [dependency build instructions](Extern/README.md). These commands build the CPU library, then enable a two-party convolution check with generated inputs.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_optiencode_conv -j8
ctest --test-dir build -R '^optiencode_conv$' --output-on-failure
```

Use `-DSEC_PPDL_BUILD_EXPERIMENTS=ON` for end-to-end models and benchmarks. Use a separate `build-gpu/` directory with `-DUSE_HE_GPU=ON` after installing the [GPU dependency](Extern/README.md#gpu-dependency).

### Run PrivCirNet

SEC-PPDL includes the two-party block-circulant linear and convolution operators. Build both CPU checks from the repository root after installing the dependencies.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target test_cir_linear test_cir_conv -j8
```

For each row, start the server in one terminal, then run the client in a second terminal on the same host. The server compares reconstructed outputs with a plaintext reference and prints a result for each case.

| Operator | Terminal 1 · server | Terminal 2 · client |
|---|---|---|
| Block-circulant linear layer | `./build/Test/test_cir_linear r=1 p=1234` | `./build/Test/test_cir_linear r=2 p=1234` |
| Block-circulant convolution | `./build/Test/test_cir_conv r=1 p=1235` | `./build/Test/test_cir_conv r=2 p=1235` |

For rotation counts and timings, use the [linear-layer benchmark](Experiments/Benchmarks/README.md). For training, layer-wise block-size selection and pretrained models, see the [official PrivCirNet repository](https://github.com/Tianshi-Xu/PrivCirNet).

## Usage

Link `LinearLayer` in your CMake application. On each party, supply an initialized HE context and a local input share. The server supplies OIHW weights and per-channel bias. The client supplies zero tensors of the same shapes.

```cpp
#include <LinearLayer/ConvOptiEncode.h>

he.GenerateNewKey(LinearLayer::Conv2DOptiEncode::GaloisElements(he.polyModulusDegree));
LinearLayer::OptiEncodeOptions options;
options.threads = 16;
LinearLayer::Conv2DOptiEncode conv(height, stride, padding, weights, bias, &he, options);
auto output_share = conv(input_share);
```

For a power-of-two plaintext modulus, set `options.ciphertext_inverse_normalization = true` on both parties. For depthwise convolution, set `options.depthwise = true` and supply weights shaped `[channels, 1, kernel, kernel]`. [Two-party example and plaintext reference](Test/support/OptiEncodeCases.h)

## Performance

| Experiment | What is measured |
|---|---|
| [OptiPrime ImageNet](Experiments/OptiPrime/README.md#performance) | CPU LAN/WAN latency, bidirectional traffic, and measured ResNet-18 integer plaintext and encrypted top-1/top-5 accuracy |
| [FaceResNet64](Experiments/FaceResNet/README.md) | Single-image embedding and two-image verification; fixed-point backbones and Compact W4A8 drivers |
| [Operator benchmarks](Experiments/Benchmarks/README.md) | Per-call timing and traffic for encrypted convolutions, and block-circulant matrix operation counts |
| [Falcon depthwise comparison](Experiments/OptiPrime/falcon/README.md) | Layer-level packing, operation counts and MobileNetV2 comparison |

Each experiment documents its own network settings, timing boundaries and units.

## Publications

### MICRO 2026 · OptiPrime

OptiPrime's released inference code uses SEC-PPDL's OptiEncode convolution. The ImageNet experiment shows how to compose the operator with MPC activations.

> Jiangrui Yu, Ye Yu, Si Chen, Chenqi Lin, Wenxuan Zeng, Junfeng Fan, Mingyu Gao, and Meng Li. **OptiPrime: Optimizing Private Inference through Protocol-Hardware Co-design.** MICRO 2026.

[Paper](https://arxiv.org/abs/2609.16898) · [Project](https://jerryfishy.github.io/OptiPrime/) · [Operator](Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h) · [Evaluation](Experiments/OptiPrime/README.md) · [BibTeX](#citation)

### USENIX Security 2025 · Breaking the Layer Barrier

The repository includes CKKS encoding primitives and linear-operator checks relevant to hybrid CKKS/MPC inference.

> Tianshi Xu, Wen-jie Lu, Jiangrui Yu, Yi Chen, Chenqi Lin, Runsheng Wang, and Meng Li. **Breaking the Layer Barrier: Remodeling Private Transformer Inference with Hybrid CKKS and MPC.** USENIX Security 2025.

[Paper](https://www.usenix.org/conference/usenixsecurity25/presentation/xu-tianshi) · [Primitives](Operator/LinearOperator/include/LinearOperator/Polynomial.h) · [Tests](Test/src/TestLinearOperator.cpp) · [BibTeX](#citation)

### NeurIPS 2024 · PrivCirNet

The authors distribute their private-inference operators through SEC-PPDL. The original project provides training code and pretrained networks.

> Tianshi Xu, Lemeng Wu, Runsheng Wang, and Meng Li. **PrivCirNet: Efficient Private Inference via Block Circulant Transformation.** NeurIPS 2024.

[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ca9873918aa72e9033041f76e77b5c15-Abstract-Conference.html) · [Project and models](https://github.com/Tianshi-Xu/PrivCirNet) · [Run the operators](#run-privcirnet) · [BibTeX](#citation)

### Comparison methods

- **BumbleBee (NDSS 2025)** — shared-input matrix multiplication with HE cross terms. [Paper](https://doi.org/10.14722/ndss.2025.230057) · [Implementation](Layer/LinearLayer/src/Linear.cpp)
- **BOLT (IEEE S&P 2024)** — packed linear layer. [Paper](https://doi.org/10.1109/SP54263.2024.00130) · [Implementation](Layer/LinearLayer/src/Linear.cpp)
- **Falcon (ICCAD 2023)** — depthwise packing reproduction and MobileNetV2 comparison. [Paper](https://arxiv.org/abs/2308.13189) · [Reproduction](Experiments/OptiPrime/falcon/README.md)
- **Cheetah (USENIX Security 2022)** — encrypted convolution baseline. [Paper](https://www.usenix.org/conference/usenixsecurity22/presentation/huang-zhicong) · [Implementation](Layer/LinearLayer/src/ConvCheetah.cpp)

To add a user publication, include its venue, paper link and an artifact showing how it uses SEC-PPDL.

## License

This repository currently has no project-wide license file. Third-party components carry their own license notices. [Dependencies](Extern/README.md) · [ArgMapping license](Utils/include/Utils/ArgMapping/LICENSE)

## Citation

Please cite PrivCirNet and Breaking the Layer Barrier when using SEC-PPDL. If you use its OptiEncode convolution implementation, please also cite OptiPrime.

```bibtex
@article{xu2024privcirnet,
  title={Privcirnet: Efficient private inference via block circulant transformation},
  author={Xu, Tianshi and Wu, Lemeng and Wang, Runsheng and Li, Meng},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={111802--111831},
  year={2024}
}

@inproceedings{xu2025breaking,
  title={Breaking the layer barrier: Remodeling private transformer inference with hybrid {CKKS} and {MPC}},
  author={Xu, Tianshi and Lu, Wen-jie and Yu, Jiangrui and Chen, Yi and Lin, Chenqi and Wang, Runsheng and Li, Meng},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={2653--2672},
  year={2025}
}

@inproceedings{yu2026optiprime,
  title={{OptiPrime}: Optimizing Private Inference through Protocol-Hardware Co-design},
  author={Yu, Jiangrui and Yu, Ye and Chen, Si and Lin, Chenqi and Zeng, Wenxuan and Fan, Junfeng and Gao, Mingyu and Li, Meng},
  booktitle={Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2026},
  url={https://arxiv.org/abs/2609.16898}
}
```
