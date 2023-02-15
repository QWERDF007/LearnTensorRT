#ifndef LEARNTRT_SENET_H
#define LEARNTRT_SENET_H

#include <map>
#include <string>

#include "NvInfer.h"

namespace LearnTRT {
namespace models {

void se_resnext50_32x4d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap,
                        nvinfer1::ITensor &input, std::string lname);

} // namespace models
} // namespace LearnTRT

#endif // LEARNTRT_SENET_H