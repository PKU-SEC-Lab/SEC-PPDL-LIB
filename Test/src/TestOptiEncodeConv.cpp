#include "../support/OptiEncodeCases.h"

int main(int argc, char **argv) {
    return OptiEncodeChecks::Run(argc, argv, [](auto &layer, auto &input, auto &, size_t, int) {
        return layer(input);
    });
}
