#ifndef LEARNTRT_UTILS_H
#define LEARNTRT_UTILS_H

#include <map>
#include <ostream>
#include <string>

#include "NvInfer.h"

namespace LearnTRT {
namespace utils {

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

inline std::ostream &operator<<(std::ostream &os, nvinfer1::ITensor &tensor)
{
    auto dims = tensor.getDimensions();
    os << " [ ";
    for (int i = 0; i < dims.nbDims; ++i) {
        os << dims.d[i] << " ";
    }
    os << "] ";
    return os;
}

} // namespace utils
} // namespace LearnTRT

#endif // LEARNTRT_UTILS_H