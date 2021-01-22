#ifndef __BASIC_MODEL_HPP
#define __BASIC_MODEL_HPP

#include "../types.hpp"
#include "../json.hpp"

struct PARAM_OPTION{
	float fLRFactor = 1.f;
	float fWDFactor = 1.f;
	bool operator < (const PARAM_OPTION &b) const {
		return (b.fLRFactor == fLRFactor && b.fWDFactor < fWDFactor) || 
				(b.fLRFactor < fLRFactor);
	}
	bool operator == (const PARAM_OPTION &b) const {
		return (b.fLRFactor == fLRFactor && b.fWDFactor == fWDFactor);
	}
};

class BasicModel : public torch::nn::Module {
public:
	BasicModel() = default;

	virtual void Initialize(const nlohmann::json &jConf) = 0;

	virtual TENSOR_ARY Forward(TENSOR_ARY inputs) = 0;

	virtual PARAM_OPTION GetParamOption(const std::string &strParamName) const;

	virtual void TrainMode(bool bTrain = true);

	virtual void SetDevice(torch::Device device);

	virtual NAMED_PARAMS NamedParameters() const;

	virtual NAMED_PARAMS NamedBuffers() const;

	virtual void InitWeights(WEIGHT_INIT_PROC InitProc);

	virtual void LoadWeights(const std::string &strFilename);

	virtual void SaveWeights(const std::string &strFilename) const;
};

#endif // #ifndef __BASIC_MODEL_HPP