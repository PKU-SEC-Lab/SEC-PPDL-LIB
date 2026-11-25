# Reproduce Falcon depthwise packing

This two-party BFV microbenchmark implements the symmetric high/low filter order and zero overlap from Section IV of the ICCAD 2023 Falcon paper. It compares the result with SEC-PPDL's actual depthwise OptiEncode operator. Each run uses reproducible public W4 kernels and A8 activations, random output masks, padding 1, and an independent spatial reference for every scalar result. The shape list comes from an exported ImageNet MobileNetV2 graph. [Falcon paper](https://arxiv.org/pdf/2308.13189)

Run from the repository root with the CPU experiments enabled in CMake (`-DSEC_PPDL_BUILD_EXPERIMENTS=ON`).

```bash
cmake --build build --target test_falcon_depthwise -j8
exp=Experiments/OptiPrime/falcon
out=build/falcon-repro
mkdir -p "$out"
for n in 4096 8192; do
    python3 "$exp/plan.py" build/mbv2-bundle/graph.txt "$n" "$out/plan-$n.txt"
    python3 "$exp/run.py" build/Experiments/OptiPrime/test_falcon_depthwise \
        "$out/plan-$n.txt" "$n" "$out/n$n"
done
```

At degree 4096 the planner uses the four shapes from Fig. 11, assumes stride 1, and searches channel tilings on full padded patches using the paper's input/output polynomial count objective. At degree 8192 it uses the 17 actual depthwise layer strides and searches uniform rectangular output tiles and channel blocks for Falcon. Both parties use the same spatial tiling within each method. Pass `--match-spatial` to the 8192 planner to give Falcon the production OptiEncode tile dimensions. Each plan row contains the input height, channel count, stride, output tile rows and columns, input channels per ciphertext and filters per weight plaintext.

Falcon packs input channels in consecutive spatial planes. Neighboring weight filters share zero planes. The planner enforces the paper's capacity bound and verifies the extent of valid coefficients under negacyclic wrap. Symmetric pairing is continued into odd and partial channel groups by explicitly tracking offsets and output indices. This extends the even-group example in the paper. At degree 8192 the search weights input and output ciphertext counts by their different sizes on this transport.

Both operators use SEC-PPDL input serialization, one 50-bit active data prime, a 50-bit special prime, plaintext modulus 2^20, and compact masked outputs. The server sends all of c1 and only the required c0 coefficients. The `layers.csv` counters measure application bytes sent by each party during the operator call. Key exchange, setup, nonlinear layers, correctness openings and network headers are outside the interval. `dense_CPmult` and `dense_auto` give schedule counts for nonzero filter diagonals. This experiment uses online HE evaluation without per-image HE preprocessing. The original paper accounts for one Nq-bit input term per ciphertext; its reported MB and SEC-PPDL application bytes use different accounting.

## End-to-end MobileNetV2 comparison

Use the same exported bundle, test cases, online HE transport, CPU thread count, and nonlinear protocol in both runs. The Falcon plan replaces only the depthwise layers. Weight NTTs and output indices are prepared during model setup. Both CPU operators parallelize their per-ciphertext work and keep network I/O outside the worker pool.

```bash
cmake --build build --target test_quant_imagenet -j8
for method in optiencode falcon; do
    args=()
    if [ "$method" = falcon ]; then args+=(--falcon-plan build/falcon-repro/plan-8192.txt); fi
    python3 Experiments/OptiPrime/run.py --bundle build/mbv2-bundle --linear cpu \
        --mode exact --compact-output --threads 16 --repeat 3 --trace-count 0 \
        "${args[@]}" --output "build/falcon-repro/e2e/$method"
done
```

The runner checks all 1000 logits against the integer reference before recording each prediction. In `summary.json`, `depthwise_bytes` and `depthwise_seconds` cover the 17 depthwise operator calls within the linear component. Subtract these from the linear totals to obtain other linear work, including FC and quantization gains. Online time includes secure argmax. Offline covers fresh per-image preprocessing. Model setup and test-vector openings are outside the per-image breakdown.
