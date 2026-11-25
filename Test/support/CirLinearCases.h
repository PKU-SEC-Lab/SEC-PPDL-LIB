#pragma once
#include <LinearLayer/CirLinear.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <iostream>


namespace CirLinearChecks {

using namespace std;
using namespace LinearLayer;

/**
 * Verify correctness against pre-computed expected output.
 */
inline bool verify_result(
    const Tensor<uint64_t>& output,
    const Tensor<uint64_t>& output_peer,
    const Tensor<uint64_t>& expected,
    uint64_t plain_mod)
{
    for (uint64_t i = 0; i < output.size(); i++) {
        uint64_t recon = (output(i) + output_peer(i)) % plain_mod;
        if (recon != expected(i)) return false;
    }
    return true;
}

/**
 * Compute expected output: Y[i,j] = sum_k weight[k,j] * input[i,k]
 */
inline Tensor<uint64_t> compute_expected(
    const Tensor<uint64_t>& input,
    const Tensor<uint64_t>& weight,
    uint64_t dim_0, uint64_t dim_1, uint64_t dim_2,
    uint64_t plain_mod)
{
    Tensor<uint64_t> expected({dim_0, dim_2});
    for (uint64_t i = 0; i < dim_0; i++) {
        for (uint64_t j = 0; j < dim_2; j++) {
            __uint128_t acc = 0;
            for (uint64_t k = 0; k < dim_1; k++) {
                acc += (__uint128_t)weight({k, j}) * input({i, k});
            }
            expected({i, j}) = acc % plain_mod;
        }
    }
    return expected;
}

/**
 * Create a block circulant weight matrix (all 1s in first column).
 */
inline Tensor<uint64_t> create_block_circulant_weight(
    uint64_t dim_1, uint64_t dim_2, uint64_t block_size)
{
    Tensor<uint64_t> weight({dim_1, dim_2});
    uint64_t num_blocks_1 = dim_1 / block_size;
    uint64_t num_blocks_2 = dim_2 / block_size;

    for (uint64_t bi = 0; bi < num_blocks_1; bi++) {
        for (uint64_t bj = 0; bj < num_blocks_2; bj++) {
            for (uint64_t i = 0; i < block_size; i++) {
                for (uint64_t j = 0; j < block_size; j++) {
                    weight({bi * block_size + i, bj * block_size + j}) = 1;
                }
            }
        }
    }
    return weight;
}

template <typename Report>
int Run(int argc, char **argv, Report report) {
    int party = 0, port = 32000;
    std::string address = "127.0.0.1";
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);

    Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    HE::HEEvaluator HE(netio, party, 8192, 60, Datatype::HOST);
    HE.GenerateNewKey();

    // Benchmark dimensions
    uint64_t dim_0 = 64, dim_1 = 256, dim_2 = 256;

    bool all_passed = true;
    for (uint64_t block_size : {1, 2, 4, 8, 16, 32, 64}) {
        Tensor<uint64_t> input({dim_0, dim_1});
        Tensor<uint64_t> weight = create_block_circulant_weight(dim_1, dim_2, block_size);
        Tensor<uint64_t> bias({dim_0, dim_2});

        if (party == ALICE) {
            for (uint64_t i = 0; i < input.size(); i++) input(i) = i % 2;
        } else {
            for (uint64_t i = 0; i < input.size(); i++) input(i) = 0;
        }
        for (uint64_t i = 0; i < bias.size(); i++) bias(i) = 0;

        Tensor<uint64_t> expected = compute_expected(input, weight, dim_0, dim_1, dim_2, HE.plain_mod);

        CirLinearNest layer(dim_0, block_size, weight, bias, &HE);
        Tensor<uint64_t> output = layer(input);

        bool pass = true;
        if (party == ALICE) {
            Tensor<uint64_t> output_peer(output.shape());
            netio->recv_data(output_peer.data().data(), output_peer.size() * sizeof(uint64_t));
            pass = verify_result(output, output_peer, expected, HE.plain_mod);

            report(block_size, layer, pass);
            all_passed &= pass;
        } else {
            netio->send_data(output.data().data(), output.size() * sizeof(uint64_t));
        }
    }

    delete netio;
    return all_passed ? 0 : 1;
}
} // namespace CirLinearChecks
