#include <iostream>

#include "../../myutils.h"
#include "../layers/batchnorm.h"
#include "senet.h"

using namespace nvinfer1;
using namespace LearnTRT::utils;

namespace LearnTRT {
namespace models {

namespace {

ILayer *SEModule(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c, int w,
                 std::string lname)
{
    // add '.' to lname
    lname = lname.empty() ? lname : lname.back() == '.' ? lname : lname + ".";

    // (avg_pool): AdaptiveAvgPool2d(output_size=1)
    IPoolingLayer *avg_pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW(w, w));
    avg_pool->setStrideNd(DimsHW{w, w});
    avg_pool->setName((lname + "avg_pool").c_str());

    // (fc1): Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
    IFullyConnectedLayer *fc1 = network->addFullyConnected(
        *avg_pool->getOutput(0), c / 16, weightMap[lname + "fc.0.weight"], weightMap[lname + "fc.0.bias"]);
    fc1->setName((lname + "fc1").c_str());

    // (relu): ReLU(inplace=True)
    IActivationLayer *relu = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    // (fc2): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
    IFullyConnectedLayer *fc2 = network->addFullyConnected(*relu->getOutput(0), c, weightMap[lname + "fc.2.weight"],
                                                           weightMap[lname + "fc.2.bias"]);
    // (sigmoid): Sigmoid()
    IActivationLayer *sigmoid = network->addActivation(*fc2->getOutput(0), ActivationType::kSIGMOID);
    // scale
    ILayer *se = network->addElementWise(input, *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    return se;
}

IPoolingLayer *layer0(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input,
                      std::string lname)
{
    // add '.' to lname
    lname = lname.empty() ? lname : lname.back() == '.' ? lname : lname + ".";

    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    // (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    IConvolutionLayer *conv1 =
        network->addConvolutionNd(input, 64, DimsHW{7, 7}, weightMap[lname + "conv1.weight"], emptywts);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});
    conv1->setName((lname + "conv1").c_str());

    // (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5f);
    bn1->setName((lname + "bn1").c_str());

    // (relu1): ReLU(inplace=True)
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    // (pool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    // 这里用 ceil 代替 floor，所以要 padding 1
    IPoolingLayer *pool = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool->setStrideNd(DimsHW{2, 2});
    pool->setPaddingNd(DimsHW{1, 1});
    pool->setName((lname + "pool").c_str());
    return pool;
}

} // namespace

void se_resnext50_32x4d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap,
                        nvinfer1::ITensor &input, std::string lname)
{
    // add '.' to lname
    lname = lname.empty() ? lname : lname.back() == '.' ? lname : lname + ".";

    std::cout << "input:" << input << std::endl;
    auto pool = layer0(network, weightMap, input, lname + "layer0");
    std::cout << "layer0:" << *pool->getOutput(0) << std::endl;
}

} // namespace models
} // namespace LearnTRT