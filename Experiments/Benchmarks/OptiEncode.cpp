#include "OptiEncodeCases.h"
#include <chrono>
#include <type_traits>

int main(int argc, char **argv) {
    return OptiEncodeChecks::Run(argc, argv, [](auto &layer, auto &input, auto &he, size_t index, int call) {
        const auto bytes_before = he.IO->counter;
        const auto start = std::chrono::steady_clock::now();
        auto output = layer(input);
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        if (he.server) {
            if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, LinearLayer::Conv2DOptiEncode>) {
                const auto &stats = layer.Stats();
                std::cout << "OptiEncode case=" << index << " call=" << call
                          << " C=" << layer.Layout().channels << " tiles=" << layer.Layout().TileCount()
                          << " in_ct=" << stats.input_ciphertexts << " out_ct=" << stats.output_ciphertexts
                          << " compute=" << stats.compute_seconds << " offline=" << stats.offline_seconds;
            } else {
                std::cout << "Cheetah case=" << index << " in_ct=" << layer.dC * layer.dH * layer.dW
                          << " out_ct=" << layer.dM * layer.dH * layer.dW;
            }
            std::cout << " seconds=" << elapsed.count() << " server_bytes=" << he.IO->counter - bytes_before << '\n';
        }
        return output;
    });
}
