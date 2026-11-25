#include "CirLinearCases.h"

int main(int argc, char **argv) {
    return CirLinearChecks::Run(argc, argv, [](auto block_size, auto &layer, bool passed) {
        std::cout << "block=" << block_size << " ntt=" << layer.ntt_size << " tile=" << layer.tile_size
                  << " rotations=" << layer.getRotationCount() << " multiplies=" << layer.getMultiplyCount()
                  << " rotation_ms=" << layer.getRotationTimeMs() << " multiply_ms=" << layer.getMultiplyTimeMs()
                  << (passed ? " PASS" : " FAIL") << '\n';
    });
}
