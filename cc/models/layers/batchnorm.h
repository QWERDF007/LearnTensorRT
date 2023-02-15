#ifndef LEARNTRT_BATCHNORM_H
#define LEARNTRT_BATCHNORM_H

#include <map>
#include <string>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

namespace LearnTRT {
namespace models {

nvinfer1::IScaleLayer *addBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                      std::map<std::string, nvinfer1::Weights> &weightMap, nvinfer1::ITensor &input,
                                      std::string lname, float eps);

} // namespace models
} // namespace LearnTRT

#endif // LEARNTRT_BATCHNORM_H
