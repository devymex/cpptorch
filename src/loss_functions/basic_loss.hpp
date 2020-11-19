#ifndef __BASIC_LOSS
#define __BASIC_LOSS

#include <torch/torch.h>
#include "../json.hpp"

class BasicLoss {
public:
	virtual void Initialize(const nlohmann::json &jConf) {
	}
	virtual float Backward(torch::Tensor tOutput, torch::Tensor tTarget) = 0;
	virtual float Evaluate(torch::Tensor tOutput, torch::Tensor tTarget) = 0;
	virtual std::string FlushResults() = 0;
};

#endif // #ifndef __BASIC_LOSS