#ifndef __TYPES_HPP
#define __TYPES_HPP

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn.h>
#include <c10/cuda/CUDAStream.h>

using TENSOR_ARY = std::vector<torch::Tensor>;
using NAMED_PARAMS = std::map<std::string, torch::Tensor>;
using WEIGHT_INIT_PROC = std::function<bool(const std::string&,
		NAMED_PARAMS&, NAMED_PARAMS&)>;

// #define DEBUG_LOAD_TEST_IMG
#define DEBUG_DUMP_YOLO_DELTA
// #define DEBUG_DRAW_IMAGE_AND_LABEL


#endif // #ifndef __TYPES_HPP