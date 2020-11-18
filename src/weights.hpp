#ifndef __WEIGHTS_HPP
#define __WEIGHTS_HPP

#include <torch/script.h>

using NAMED_PARAMS = std::map<std::string, torch::Tensor>;

NAMED_PARAMS LoadWeights(const std::string &strFilename);

void SaveWeights(const NAMED_PARAMS &weights, const std::string &strFilename);

bool InitModuleWeight(const std::string &strModuleType, NAMED_PARAMS &weights);

#endif // #ifndef __WEIGHTS_HPP