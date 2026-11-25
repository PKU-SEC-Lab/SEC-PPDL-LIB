# Legacy ResNet examples

The historical fixed-point examples provide three-stage CIFAR and four-stage ImageNet model constructors. The four-stage example uses basic residual blocks and a stride-four stem. For the standard ImageNet ResNet-18/34/50 graphs and their measured results, see the [OptiPrime experiment](../OptiPrime/README.md).

Enable `SEC_PPDL_BUILD_EXPERIMENTS` and build the `test_resblock` or `test_resnet` target explicitly. The block example uses the CPU. The full legacy driver selects the device backend and needs a GPU build. Run the resulting binary in two terminals with `r=1` and `r=2` and the same `p=<port>`.
