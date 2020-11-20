#include "../json.hpp"
#include "../models/basic_model.hpp"

class BasicOptimizer {
public:
	virtual void Initialize(const nlohmann::json &jConf) = 0;
	virtual void SetModel(BasicModel &model) = 0;
	virtual void ZeroGrad() = 0;
	virtual void IterStep() = 0;
	virtual void EpochStep() = 0;
};
