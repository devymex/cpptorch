#include "../creator.hpp"
#include "../argman.hpp"
#include "basic_model.hpp"

class JITModel : public BasicModel {
public:
	void Initialize(const nlohmann::json &jConf) override {
		ArgMan argMan;
		Arg<std::string> argModelFile("model_file", argMan);
		Arg<bool> argReintialize("reinitialize", false, argMan);
		ParseArgsFromJson(jConf, argMan);
		m_JitModule = torch::jit::load(argModelFile());
		m_bReinit = argReintialize();
	}

	TENSOR_ARY Forward(TENSOR_ARY inputs) override {
		std::vector<torch::jit::IValue> jitInputs;
		for (auto i: inputs) {
			jitInputs.emplace_back(std::move(i));
		}
		auto outputs = m_JitModule.forward(jitInputs);
		TENSOR_ARY results;
		if (outputs.isTuple()) {
			auto pTuple = outputs.toTuple();
			for (auto &out : pTuple->elements()) {
				CHECK(out.isTensor());
				results.emplace_back(out.toTensor());
			}
		} else if (outputs.isTensor()) {
			results.emplace_back(outputs.toTensor());
		} else {
			LOG(FATAL) << outputs.type()->str();
		}
		return results;
	}

	void TrainMode(bool bTrain = true) override {
		m_JitModule.train(bTrain);
	}

	void SetDevice(torch::Device device) override {
		m_JitModule.to(device);
	}

	NAMED_PARAMS NamedParameters() const override {
		auto jitParams = m_JitModule.named_parameters();
		NAMED_PARAMS params;
		for (const auto &p : jitParams) {
			params[p.name] = p.value;
		}
		return params;
	}

	void InitWeights(WEIGHT_INIT_PROC InitProc) override {
		if (m_bReinit) {
			for (const auto &subMod: m_JitModule.named_children()) {
				auto typeName = subMod.value.type()->name();
				if (typeName.has_value()) {
					NAMED_PARAMS namedParams;
					for (const auto &params: subMod.value.named_parameters()) {
						namedParams[params.name] = params.value;
					}
					if (!InitProc(typeName->name(), namedParams)) {
						for (const auto &param: namedParams) {
							LOG(INFO) << "Unintialized parameter: " << param.first;
						}
					}
				}
			}
		}
	}

private:
	torch::jit::Module m_JitModule;
	bool m_bReinit = false;
};

REGISTER_CREATOR(BasicModel, JITModel, "JIT");
