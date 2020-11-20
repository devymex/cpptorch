#include <map>
#include "../argman.hpp"
#include "../creator.hpp"
#include "basic_optimizer.hpp"

class SGDOptimizer : public BasicOptimizer {
public:
	SGDOptimizer() : m_DefaultOpt(0.) {
	}

	void Initialize(const nlohmann::json &jConf) {
		LOG(INFO) << jConf.dump(4);
		ArgMan argMan;
		Arg<float> argLearningRate("learning_rate", 0.01f, argMan);
		Arg<float> argWeightDecay("weight_decay", 0.f, argMan);
		Arg<float> argMomentum("momentum", 0.f, argMan);
		Arg<uint64_t> argLRStepSize("lr_step_epochs", 0, argMan);
		Arg<float> argLRStepGamma("lr_step_gamma", 0.f, argMan);
		ParseArgsFromJson(jConf, argMan);

		m_DefaultOpt = torch::optim::SGDOptions(argLearningRate())
			.weight_decay(argWeightDecay())
			.momentum(argMomentum());
	}

	void SetModel(BasicModel &model) override {
		std::map<PARAM_OPTION, std::vector<torch::Tensor>> optionParams;
		for (const auto &param : model.named_parameters()) {
			PARAM_OPTION opt = model.GetParamOption(param.key());
			optionParams[opt].emplace_back(param.value());
		}
		std::vector<torch::optim::OptimizerParamGroup> optimizeGroups;
		for (auto &group : optionParams) {
			std::unique_ptr<torch::optim::SGDOptions> pOpt(
					new torch::optim::SGDOptions(m_DefaultOpt));
			pOpt->lr() *= group.first.fLRFactor;
			pOpt->weight_decay() *= group.first.fWDFactor;
			optimizeGroups.emplace_back(std::move(group.second), std::move(pOpt));
		}
		m_pSGD.reset(new torch::optim::SGD(optimizeGroups, m_DefaultOpt));
	}

	void ZeroGrad() override {
		m_pSGD->zero_grad();
	}

	void IterStep() override {
		m_pSGD->step();
	}

	void EpochStep() override {
	}
private:
	torch::optim::SGDOptions m_DefaultOpt;
	std::unique_ptr<torch::optim::SGD> m_pSGD;
};

REGISTER_CREATOR(BasicOptimizer, SGDOptimizer, "SGD");
