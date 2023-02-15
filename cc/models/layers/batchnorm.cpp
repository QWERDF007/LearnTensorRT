#include "batchnorm.h"
#include <cmath>

using namespace nvinfer1;

namespace LearnTRT {
namespace models {

nvinfer1::IScaleLayer *addBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                      std::map<std::string, nvinfer1::Weights> &weightMap, nvinfer1::ITensor &input,
                                      std::string lname, float eps)
{
    float *gamma = (float *)weightMap[lname + ".weight"].values;
    float *beta = (float *)weightMap[lname + ".bias"].values;
    float *mean = (float *)weightMap[lname + ".running_mean"].values;
    float *var = (float *)weightMap[lname + ".running_var"].values;
    int64_t len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    return scale_1;
}

} // namespace models
} // namespace LearnTRT