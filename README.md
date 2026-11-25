![SEC-PPDL by PKU SEC Lab: homomorphic convolution followed by an MPC activation](docs/assets/sec-ppdl.jpg)

<h3 align="center">SEC-PPDL: A High-Performance Library for Two-Party Private Neural Network Inference on CPUs and GPUs</h3>

<p align="center">An easy-to-use hybrid FHE–MPC framework for high-performance private neural network inference.</p>

<p align="center">
  <a href="https://github.com/PKU-SEC-Lab/SEC-PPDL-LIB/tree/v1.0"><img src="https://img.shields.io/badge/version-1.0-38689B" alt="Version: 1.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-29736B" alt="License: MIT"></a>
  <a href="#current-support"><img src="https://img.shields.io/badge/Operator-examples-685394" alt="Operator examples"></a>
  <a href="#experiments-and-results"><img src="https://img.shields.io/badge/Experiment-results-9A591F" alt="Experiment results"></a>
  <a href="https://mengli.me/"><img src="https://img.shields.io/badge/PKU-SEC_Lab-9B354C" alt="PKU SEC Lab homepage"></a>
</p>

## About

SEC-PPDL is an open-source C++ library from PKU SEC Lab for evaluating neural networks while keeping the client’s input and the server’s model private. It combines homomorphic encryption for convolution and matrix multiplication with secure two-party protocols for activations and other nonlinear operations.

You can integrate individual layers into a C++ application, check their outputs against plaintext, or measure latency and network traffic for ResNet, VGG, and MobileNetV2. The same OptiEncode interface supports dense and depthwise convolution on CPUs and CUDA GPUs. [Operator API](#usage) · [Experiments and results](#experiments-and-results)

## Contents

1. [News](#news)
2. [Current support](#current-support)
3. [Quick start](#quick-start)
4. [Operator API](#usage)
5. [Repository layout](#repository-layout)
6. [Experiments and results](#experiments-and-results)
7. [Publications](#publications)
8. [License](#license) and [citation](#citation)
9. [Contributing](CONTRIBUTING.md)

## News

- 📦 **2026-11-26 · SEC-PPDL 1.0.** The MIT-licensed release includes CPU and CUDA operators, two-party checks and end-to-end experiments. [Version 1.0](https://github.com/PKU-SEC-Lab/SEC-PPDL-LIB/tree/v1.0)
- 📄 **MICRO 2026 · OptiPrime.** OptiEncode convolution runs on CPU and CUDA. A subsequent extension supports packed depthwise convolution for MobileNetV2. [Project page](https://jerryfishy.github.io/OptiPrime/) · [Depthwise layout](Experiments/OptiPrime/depthwise.md)

## Current support

| Component | Available in the library | API | Run an example |
|---|---|---|---|
| Encrypted convolution | OptiEncode dense and multiplier-one depthwise convolution on CPU and CUDA; Cheetah and block-circulant convolution | [OptiEncode](Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h) · [Other convolutions](Layer/LinearLayer/include/LinearLayer/Conv.h) | [OptiEncode convolution](#quick-start) |
| Linear algebra | Packed matrix multiplication and block-circulant linear layers | [Linear](Layer/LinearLayer/include/LinearLayer/Linear.h) · [CirLinear](Layer/LinearLayer/include/LinearLayer/CirLinear.h) | [Block-circulant layers](Experiments/PrivCirNet/README.md#build-the-checks) |
| Secure nonlinear operators | ReLU, pooling, fixed-point comparison, selection and truncation | [Layers](Layer/NonlinearLayer/include/NonlinearLayer) · [Fixed-point protocols](Operator/NonlinearOperator/include/NonlinearOperator/FixPoint.h) | [Fixed-point comparison](#two-party-operator-checks) |
| Two-party runtime | HE key setup, IKNP or Ferret OT, communication counters | [Runtime](Runtime/include/Runtime/Primitive.h) | [Two-party setup](#two-party-operator-checks) |

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

### Two-party operator checks

With `BUILD_TESTING=ON`, build the checks from the repository root.

```bash
cmake --build build --target test_linear_operator test_fixpoint -j8
```

Start the server in one terminal, then run its client in a second terminal on the same machine.

| Check | Server | Client |
|---|---|---|
| Element-wise square and product | `./build/Test/test_linear_operator r=1 p=32000` | `./build/Test/test_linear_operator r=2 p=32000` |
| Fixed-point comparison with a constant | `./build/Test/test_fixpoint r=1 p=32001` | `./build/Test/test_fixpoint r=2 p=32001` |

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

## Repository layout

| Directory | What to look for |
|---|---|
| [Operator/](Operator), [Layer/](Layer) | Encoding and fixed-point primitives, plus callable layers such as convolution, ReLU and pooling. Each module exposes its public headers under `include/`. |
| [HE/](HE), [OT/](OT), [Runtime/](Runtime) | Encryption backends, oblivious transfer protocols and shared two-party setup. |
| [Datatype/](Datatype), [Utils/](Utils), [Extern/](Extern) | Tensor types, networking and threading utilities, and pinned third-party dependencies. |
| [Test/](Test) | Operator correctness checks and their plaintext references. |
| [Experiments/](Experiments) | Paper-specific guides, end-to-end models, benchmarks and measurement summaries. |

See the [contributor guide](CONTRIBUTING.md#directory-ownership) for file placement and CMake conventions.

<a name="performance"></a>

## Experiments and results

| Experiment | What is measured |
|---|---|
| [OptiPrime ImageNet](Experiments/OptiPrime/README.md#performance) | CPU LAN/WAN latency, bidirectional traffic, and measured ResNet-18 integer plaintext and encrypted top-1/top-5 accuracy |
| [FaceResNet64](Experiments/FaceResNet/README.md) | Single-image embedding and two-image verification; fixed-point backbones and Compact W4A8 drivers |
| [PrivCirNet](Experiments/PrivCirNet/README.md) | Block-circulant linear and convolution outputs checked against plaintext; linear-layer rotation and multiplication counts and timings |
| [Operator benchmarks](Experiments/Benchmarks/README.md) | Per-call timing and traffic for encrypted convolutions, and block-circulant matrix operation counts |
| [Falcon depthwise comparison](Experiments/OptiPrime/falcon/README.md) | Layer-level packing, operation counts and MobileNetV2 comparison |

Each experiment documents its own network settings, timing boundaries and units.

## Publications

### MICRO 2026 · OptiPrime

We implement OptiEncode convolution on CPUs and GPUs. In the ImageNet experiment, we combine it with MPC activations and measure end-to-end latency and communication.

> Jiangrui Yu, Ye Yu, Si Chen, Chenqi Lin, Wenxuan Zeng, Junfeng Fan, Mingyu Gao, and Meng Li. **OptiPrime: Optimizing Private Inference through Protocol-Hardware Co-design.** MICRO 2026.

[Paper](https://arxiv.org/abs/2609.16898) · [Code](Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h) · [Reproduce](Experiments/OptiPrime/README.md#quick-start) · [BibTeX](#cite-optiprime) · [Project](https://jerryfishy.github.io/OptiPrime/)

### USENIX Security 2025 · Breaking the Layer Barrier

We study how to combine CKKS and MPC for private Transformer inference. Here we provide CKKS encoding primitives and a two-party check for element-wise multiplication.

> Tianshi Xu, Wen-jie Lu, Jiangrui Yu, Yi Chen, Chenqi Lin, Runsheng Wang, and Meng Li. **Breaking the Layer Barrier: Remodeling Private Transformer Inference with Hybrid CKKS and MPC.** USENIX Security 2025.

[Paper](https://www.usenix.org/conference/usenixsecurity25/presentation/xu-tianshi) · [Code](Operator/LinearOperator/include/LinearOperator/Polynomial.h) · [Reproduce (operator check)](#two-party-operator-checks) · [BibTeX](#cite-breaking)

### NeurIPS 2024 · PrivCirNet

We implement block-circulant linear and convolution layers and check their outputs against plaintext. We provide training code and pretrained models in the PrivCirNet project.

> Tianshi Xu, Lemeng Wu, Runsheng Wang, and Meng Li. **PrivCirNet: Efficient Private Inference via Block Circulant Transformation.** NeurIPS 2024.

[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ca9873918aa72e9033041f76e77b5c15-Abstract-Conference.html) · [Code](Layer/LinearLayer/src/CirLinear.cpp) · [Reproduce (operators)](Experiments/PrivCirNet/README.md) · [BibTeX](#cite-privcirnet) · [Project and models](https://github.com/Tianshi-Xu/PrivCirNet)

### Additional implementations

- **BumbleBee (NDSS 2025)** — Matrix multiplication on secret-shared inputs, with HE for the cross terms. [Paper](https://doi.org/10.14722/ndss.2025.230057) · [Implementation](Layer/LinearLayer/src/Linear.cpp)
- **BOLT (IEEE S&P 2024)** — Packed linear layer. [Paper](https://doi.org/10.1109/SP54263.2024.00130) · [Implementation](Layer/LinearLayer/src/Linear.cpp)
- **Falcon (ICCAD 2023)** — Reproduction of the depthwise convolution packing scheme, with a MobileNetV2 comparison. [Paper](https://arxiv.org/abs/2308.13189) · [Reproduction](Experiments/OptiPrime/falcon/README.md)
- **Cheetah (USENIX Security 2022)** — Encrypted convolution used as the baseline in our inference experiments. [Paper](https://www.usenix.org/conference/usenixsecurity22/presentation/huang-zhicong) · [Implementation](Layer/LinearLayer/src/ConvCheetah.cpp)

## License

SEC-PPDL is distributed under the [MIT License](LICENSE). Third-party dependencies and files with their own notices retain their respective licenses. [Dependencies](Extern/README.md) · [ArgMapping license](Utils/include/Utils/ArgMapping/LICENSE)

## Citation

If you use SEC-PPDL in your research, please cite all three papers: PrivCirNet, Breaking the Layer Barrier, and OptiPrime.

<a name="cite-privcirnet"></a>

```bibtex
@article{xu2024privcirnet,
  title={Privcirnet: Efficient private inference via block circulant transformation},
  author={Xu, Tianshi and Wu, Lemeng and Wang, Runsheng and Li, Meng},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={111802--111831},
  year={2024}
}
```

<a name="cite-breaking"></a>

```bibtex
@inproceedings{xu2025breaking,
  title={Breaking the layer barrier: Remodeling private transformer inference with hybrid {CKKS} and {MPC}},
  author={Xu, Tianshi and Lu, Wen-jie and Yu, Jiangrui and Chen, Yi and Lin, Chenqi and Wang, Runsheng and Li, Meng},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={2653--2672},
  year={2025}
}
```

<a name="cite-optiprime"></a>

```bibtex
@inproceedings{yu2026optiprime,
  title={{OptiPrime}: Optimizing Private Inference through Protocol-Hardware Co-design},
  author={Yu, Jiangrui and Yu, Ye and Chen, Si and Lin, Chenqi and Zeng, Wenxuan and Fan, Junfeng and Gao, Mingyu and Li, Meng},
  booktitle={Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2026},
  url={https://arxiv.org/abs/2609.16898}
}
```
