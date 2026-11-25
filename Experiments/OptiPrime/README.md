# OptiPrime in SEC-PPDL

## About

[OptiEncode convolution](https://jerryfishy.github.io/OptiPrime/) is the encrypted convolution operator from OptiPrime (MICRO 2026). Link the core `LinearLayer` library to use the operator with your own encoded tensors. The optional ImageNet experiment in this directory demonstrates integer-only inference. [Paper](https://arxiv.org/abs/2609.16898)

## News

- The implementation now supports packed depthwise convolution on CPU and GPU, including all 17 depthwise layers in MobileNetV2. [Protocol and layout](depthwise.md)
- OptiPrime appeared at the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO 2026).

## Current Support

| Component | Support |
|---|---|
| Convolution | Dense and multiplier-one depthwise convolution, stride and padding, multithreaded CPU and CUDA GPU. |
| ImageNet models | ResNet-18/34/50, VGG-16/19 and MobileNetV2. [Network definitions](imagenet/architecture.py) |
| Integer inference | Exact requantization or requantization with randomized truncation. Configurable convolution weight width (2–8 bits), activation width (2–16), classifier width (2–8) and work-ring width (20–32). |
| Execution | IKNP or Ferret OT, per-image preprocessing, CUDA Graphs and optional compact HE outputs. See `python3 Experiments/OptiPrime/run.py --help` for switches. |

## Quick Start

Run from the SEC-PPDL root after [building the C++ dependencies](../../Extern/README.md). The CPU convolution test uses generated inputs and requires no model weights.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSEC_PPDL_BUILD_QUANT_EXPERIMENTS=ON
cmake --build build --target test_optiencode_conv test_quant_imagenet -j8
ctest --test-dir build -R '^optiencode_conv$' --output-on-failure
```

For ResNet-18, install PyTorch, torchvision, NumPy and Pillow. Under `build/real-rn18`, place the extracted [Imagenette calibration set](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) (`imagenette2-320/`), [ImageNetV2 matched-frequency set](https://huggingface.co/datasets/vaishaal/ImageNetV2) (`imagenetv2-matched-frequency-format-val/`) and [class index](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json). The exporter uses AIMET weights for ResNet-18 and torchvision V1 weights for the other models.

```bash
exp=Experiments/OptiPrime
python3 "$exp/model_assets.py" --output build/model-assets --weights
python3 "$exp/imagenet/evaluate.py" --model resnet18 --assets build/model-assets --data build/real-rn18 \
  --output build/rn18-w4a8 --weight-bits 4 --activation-bits 8 --fc-bits 8 --secure-images 2 --export-only
python3 "$exp/run.py" --bundle build/rn18-w4a8 --linear cpu --mode approx --threads 16 \
  --repeat 2 --trace-count 2 --output build/rn18-approx
python3 "$exp/imagenet/check_secure.py" --assets build/model-assets --results build/rn18-approx --device cpu
```

Use `--trace-count 0` for timing. For GPU inference, configure `build-gpu` with the same options and `-DUSE_HE_GPU=ON`, build `test_quant_imagenet`, and run with `--linear gpu`.

## Usage

Link `LinearLayer` and include the [public convolution header](../../Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h). The HE context selects the backend. Set `options.depthwise = true` and provide compact `[channels, 1, kernel, kernel]` weights for depthwise convolution.

```cpp
#include <LinearLayer/ConvOptiEncode.h>

he.GenerateNewKey(LinearLayer::Conv2DOptiEncode::GaloisElements(he.polyModulusDegree));
LinearLayer::OptiEncodeOptions options;
options.threads = 16;
LinearLayer::Conv2DOptiEncode conv(height, stride, padding, weights, bias, &he, options);
auto output_share = conv(input_share);
```

For the complete ImageNet experiment, link the `OptiPrime` target and use `<OptiPrime/Bundle.h>`. The [two-party runner](src/TestImageNet.cpp) shows model loading, OT preparation, inference and secure argmax.

### Compact FaceResNet64 verification

The CPU benchmark quantizes synthetic dense weights to W4 and uses A8 activations with a 20-bit accumulator. It checks the 128-dimensional embeddings and the secure cosine decision for two synthetic pairs against an integer reference. The fixture uses two channel-scale groups. This benchmark measures protocol correctness and latency on synthetic inputs.

From the repository root, use Python 3.11 or later with NumPy installed.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSEC_PPDL_BUILD_QUANT_EXPERIMENTS=ON
cmake --build build --target face_w4a8 -j8
python3 Experiments/OptiPrime/face/run.py --binary build/Experiments/OptiPrime/face_w4a8 --mode exact
python3 Experiments/OptiPrime/face/run.py --binary build/Experiments/OptiPrime/face_w4a8 --mode light --wan
```

Both modes support loopback and the optional `--wan` proxy (400 Mb/s per direction, 20 ms RTT). The runner uses 16 HE workers and three OT threads per party. Use `--server-cpus` and `--client-cpus` for optional CPU pinning. Each run writes its generated fixtures, logs and summary to a new directory under `build/face-w4a8/`. Pair latency includes both forwards and the secure match. Setup, weight packing and opening embeddings for verification are timed separately or outside that interval.

## Performance

### CPU latency and communication

Both parties ran on one AMD EPYC 9J14 host with 16 CPU workers per party. Linux `tc` shaped Docker loopback to 3,072 Mbit/s per direction and 0.3 ms RTT on LAN, or 352 Mbit/s and 40 ms RTT on WAN. Latency includes fresh per-image Ferret OT preprocessing and online inference through secure argmax. HE ran online. The timed interval excludes model setup, key exchange, OT planning and verification openings. Communication sums application bytes sent in both directions, in decimal GB.

Cheetah used fixed-point weights and inputs with eight fractional bits, a 50-bit HE plaintext prime and a 60-bit MPC ring. OptiPrime used W4A8 convolutions, an 8-bit final classifier and integer nonlinear operations with randomized truncation. Its work ring was 26 bits for VGG-19 and 20 bits for the other models. The table reports per-image means over matched inputs. [Per-image records](measurements/cpu-inference.json)

| Model | Network | Images | Cheetah (s) | OptiPrime (s) | Speedup | Cheetah (GB) | OptiPrime (GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | LAN | 16 | 45.31 | 6.77 | 6.70× | 6.938 | 0.299 |
| ResNet-18 | WAN | 16 | 235.41 | 39.79 | 5.92× | 6.938 | 0.299 |
| ResNet-34 | LAN | 1 | 76.34 | 13.29 | 5.74× | 9.934 | 0.431 |
| ResNet-50 | LAN | 1 | 595.30 | 25.51 | 23.34× | 131.848 | 1.134 |
| VGG-16 | LAN | 1 | 142.66 | 24.16 | 5.91× | 15.852 | 2.051 |
| VGG-19 | LAN | 1 | 140.10 | 44.51 | 3.15× | 17.569 | 3.759 |
| MobileNetV2 | LAN | 1 | 152.21 | 20.30 | 7.50× | 26.519 | 1.065 |

The [project-page chart](https://jerryfishy.github.io/OptiPrime/#communication) uses a separate single-image ResNet-18 observation (6.955 GB for Cheetah and 0.308 GB for OptiPrime). The demo uses the 16-image runs above. The MobileNetV2 baseline implements depthwise convolution with diagonal weights in Cheetah's dense operator.

### W4A8 integer-only results

ResNet-18 on the same 3,000 distinct ImageNetV2 matched-frequency images across 970 classes. Both runs used integer requantization with randomized truncation and an 8-bit final classifier.

| Execution | Top-1 | Top-5 |
|---|---:|---:|
| Integer plaintext | 54.70% (1,641/3,000) | 77.50% (2,325/3,000) |
| Two-party encrypted inference | 54.60% (1,638/3,000) | 77.43% (2,323/3,000) |

## Citation

```bibtex
@inproceedings{yu2026optiprime,
  title={{OptiPrime}: Optimizing Private Inference through Protocol-Hardware Co-design},
  author={Yu, Jiangrui and Yu, Ye and Chen, Si and Lin, Chenqi and Zeng, Wenxuan and Fan, Junfeng and Gao, Mingyu and Li, Meng},
  booktitle={Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2026},
  url={https://arxiv.org/abs/2609.16898}
}
```
