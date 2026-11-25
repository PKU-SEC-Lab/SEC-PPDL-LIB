#include "../support/CirLinearCases.h"

int main(int argc, char **argv) {
    return CirLinearChecks::Run(argc, argv, [](auto block_size, auto &, bool passed) {
        std::cout << "block=" << block_size << (passed ? " PASS" : " FAIL") << '\n';
    });
}
