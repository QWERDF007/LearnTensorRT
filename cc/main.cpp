#include <fstream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "logging.h"
#include "models/backbones/senet.h"
#include "myutils.h"

using namespace nvinfer1;

int main(int argc, char **argv)
{
    Logger gLogger;

    std::string weight_file{"F:/workspace/LearnTensorRT/models/"
                            "yijiaofupian640_0510_best_model.wts"};
    auto weights_map = LearnTRT::utils::loadWeights(weight_file);
    std::cout << "Create builder" << std::endl;
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    // uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(0U);

    const int INPUT_H = 640;
    const int INPUT_W = 640;
    std::cout << "Create input" << std::endl;
    ITensor *data = network->addInput("input", DataType::kFLOAT, Dims3{3, INPUT_H, INPUT_W});

    std::cout << "Create se_resnext50_32x4d" << std::endl;
    LearnTRT::models::se_resnext50_32x4d(network, weights_map, *data, "encoder");

    builder->destroy();
    config->destroy();
    return 0;
}