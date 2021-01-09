#ifndef __BASIC_LOSS
#define __BASIC_LOSS

#include "../types.hpp"
#include "../json.hpp"

class BasicLoss {
public:
	virtual void Initialize(const nlohmann::json &jConf) {
	}
	virtual float Backward(TENSOR_ARY tOutput, TENSOR_ARY tTarget) = 0;
	virtual float Evaluate(TENSOR_ARY tOutput, TENSOR_ARY tTarget) = 0;
	virtual std::string FlushResults() = 0;
};

#endif // #ifndef __BASIC_LOSS