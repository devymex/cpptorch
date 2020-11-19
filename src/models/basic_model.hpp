#ifndef __BASIC_MODEL_HPP
#define __BASIC_MODEL_HPP

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn.h>
#include "../json.hpp"

using NAMED_PARAMS = std::map<std::string, torch::Tensor>;

using WEIGHT_INIT_PROC = std::function<bool(const std::string&, NAMED_PARAMS&)>;

class BasicModel : protected torch::nn::Module {
public:
	BasicModel() = default;

	virtual void Initialize(const nlohmann::json &jConf) = 0;

	virtual torch::Tensor Forward(std::vector<torch::Tensor> inputs) = 0;

	virtual void TrainMode(bool bTrain = true);

	virtual void SetDevice(torch::Device device);
	
	virtual NAMED_PARAMS NamedParameters() const;

	virtual void InitWeights(WEIGHT_INIT_PROC InitProc);

	virtual void LoadWeights(const std::string &strFilename);

	virtual void SaveWeights(const std::string &strFilename) const;
};

#endif // #ifndef __BASIC_MODEL_HPP