# Contributing to SEC-PPDL

## Directory ownership

| Directory | What belongs here |
|---|---|
| `Datatype/` | Public tensor and backend data types. The header-only `Datatype` target exports the includes and its SEAL dependency. |
| `HE/`, `OT/` | HE contexts, ciphertext backends, OT primitives and protocols. CUDA sources live with the backend that owns them. |
| `Utils/` | Shared networking, threading and arithmetic utilities. |
| `Operator/` | Reusable linear and nonlinear operator primitives, including encoding, conversion and fixed-point protocols. |
| `Layer/` | Composable layers such as convolution, ReLU and pooling. Keep CPU and CUDA implementations next to their layer. |
| `Runtime/` | Shared two-party setup combining HE, OT and layer services. Link the header-only `Runtime` target for `<Runtime/Primitive.h>`. |
| `Test/src/`, `Test/support/` | Correctness checks and reusable plaintext references or deterministic cases. A benchmark may reuse a test fixture. |
| `Experiments/OptiPrime/` | ImageNet graph assembly, model export, inference drivers, published measurement summaries and the Falcon comparison. |
| `Experiments/FaceResNet/` | Face embeddings, cosine matching, Q12 drivers, W4A8 fixtures and evaluation. |
| `Experiments/PrivCirNet/` | Reproduction instructions for block-circulant inference operators and their benchmarks. |
| `Experiments/ResNet/` | Historical fixed-point ResNet examples. |
| `Experiments/Benchmarks/`, `Experiments/Nonlinear/` | Operator performance drivers and numerical studies. |
| `Experiments/tools/` | Network shaping and other tools shared across experiments. |
| `Extern/` | Pinned third-party sources, their patches and dependency build instructions. |
| `docs/assets/` | Versioned illustrations used by repository documentation. |
| `build/`, `build-gpu/` | Local builds, model weights, datasets, logs, traces and generated plots. |

## C++ interfaces and dependencies

- Put public headers in `<module>/include/<namespace>/`. Include them by public name, for example `<LinearLayer/ConvOptiEncode.h>`. Put implementation files, CUDA kernels and private helper headers in `src/`.
- Use C++17. Keep template definitions visible at instantiation, and use `inline` for non-template function definitions in shared headers. Keep namespace imports scoped. Follow the adjacent operator's naming conventions for API compatibility.
- Keep dataset loading, network topology, command-line parsing, benchmarking and result presentation in the owning experiment. Core targets can depend on other core targets. Experiments and tests link the core targets they exercise. Headers under an experiment’s own include directory belong to that experiment target.
- Document operator input shapes, party ownership, modulus, signedness and scaling. When packing changes, check padding, strides and partial channel blocks against a plaintext reference on both supported backends.

## CMake and validation

Libraries expose dependencies with `PUBLIC` when consumers need them through public headers. Use `PRIVATE` for implementation details and executable dependencies. Use `INTERFACE` for header-only targets. List new sources explicitly and keep programs with a `main()` in their own executable targets.

| Configuration | Command from the repository root |
|---|---|
| CPU library | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` |
| Correctness tests | `cmake -S . -B build -DBUILD_TESTING=ON` |
| Models and benchmarks | `cmake -S . -B build -DSEC_PPDL_BUILD_EXPERIMENTS=ON` |
| CUDA plus tests | `cmake -S . -B build-gpu -DUSE_HE_GPU=ON -DBUILD_TESTING=ON` |

These options persist in the selected build directory. Use a fresh directory for library-only validation. The existing `SEC_PPDL_BUILD_QUANT_EXPERIMENTS` option continues to enable the OptiPrime and FaceResNet programs. Individual benchmark targets can be built on demand.

Build with `cmake --build build -j8`. Run the configured regression suite with `ctest --test-dir build --output-on-failure`. For changes to public headers or linkage, also compile a small downstream application with `add_subdirectory` and link the documented target. For end-to-end changes, run the experiment's reference check and report the tested backend.

## Experiments and documentation

An experiment README gives its build target, commands from the repository root, dependencies, inputs and correctness reference. Published performance results identify the hardware, threads, bandwidth, RTT, measured interval, preprocessing and communication units. Keep aggregate results used by the documentation alongside the experiment. Send per-run outputs to a build directory.

Link each supported paper to its implemented component. A downstream publication entry should link a paper or public artifact identifying the library. Keep BibTeX in the root README for the library's cited works.
