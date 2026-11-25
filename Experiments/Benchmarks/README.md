# Operator benchmarks

Configure from the repository root and build the CPU benchmark targets on demand.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSEC_PPDL_BUILD_EXPERIMENTS=ON
cmake --build build --target bench_optiencode bench_cir_linear -j8
python3 Test/run_optiencode.py --binary build/Experiments/Benchmarks/bench_optiencode --bench 1 --threads 4 --repeat 1 --logs build/bench-optiencode
```

The runner checks both processes and stores per-party logs. The server reports per-call elapsed time and bytes sent before its correctness opening. The Cheetah comparison runs on the last dense convolution case. For block-circulant linear performance, launch `build/Experiments/Benchmarks/bench_cir_linear` in two terminals with `r=1 p=1234` and `r=2 p=1234`. Its table reports rotation and multiplication counts and times by block size.

The `test_HE` and `testFFN` microbenchmark targets are available in the GPU build. The [ImageNet](../OptiPrime/README.md) and [face verification](../FaceResNet/README.md) experiments measure full networks.
