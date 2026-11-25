#include <Datatype/Tensor.h>

int main() {
    Datatype::Tensor<double> tensor({2, 3}, {1., 2., 3., 4., 5., 6.});
    return tensor.size() == 6 && tensor({1, 2}) == 6. ? 0 : 1;
}
