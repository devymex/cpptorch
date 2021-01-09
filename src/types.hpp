#ifndef __TYPES_HPP
#define __TYPES_HPP

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn.h>

using TENSOR_ARY = std::vector<torch::Tensor>;
using NAMED_PARAMS = std::map<std::string, torch::Tensor>;
using WEIGHT_INIT_PROC = std::function<bool(const std::string&, NAMED_PARAMS&)>;

#endif // #ifndef __TYPES_HPP