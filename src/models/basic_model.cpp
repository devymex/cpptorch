#include "basic_model.hpp"
#include "../weights.hpp"

void BasicModel::TrainMode(bool bTrain) {
	torch::nn::Module::train(bTrain);
}

void BasicModel::SetDevice(torch::Device device) {
	this->to(device);
}

NAMED_PARAMS BasicModel::NamedParameters() const {
	NAMED_PARAMS namedParams;
	for (const auto &params : torch::nn::Module::named_parameters()) {
		namedParams[params.key()] = params.value();
	}
	return namedParams;
}

PARAM_OPTION BasicModel::GetParamOption(const std::string &strParamName) const {
	return PARAM_OPTION();
}

void BasicModel::InitWeights(WEIGHT_INIT_PROC InitProc) {
	for (const auto &pSubMod: modules()) {
		if (pSubMod->children().size() == 0) {
			NAMED_PARAMS params;
			for (auto &param: pSubMod->named_parameters()) {
				params[param.key()] = param.value();
			}
			NAMED_PARAMS buffers;
			for (const auto &buf: pSubMod->named_buffers()) {
				buffers[buf.key()] = buf.value();
			}
			if (!params.empty() || !buffers.empty()) {
				if (!InitProc(pSubMod->name(), params, buffers)) {
					LOG(INFO) << "Unintialized module: " << pSubMod->name();
				}
			}
		}
	}
}

void BasicModel::LoadWeights(const std::string &strFilename) {
	auto loadedParams = ::LoadWeights(strFilename);
	for (const auto &param: NamedParameters()) {
		auto iLoaded = loadedParams.find(param.first);
		if (iLoaded != loadedParams.end()) {
			auto paramDev = param.second.options().device();
			param.second.set_data(iLoaded->second.to(paramDev));
		} else {
			LOG(INFO) << "Unintialized parameter: " << param.first;
		}
	}
}

void BasicModel::SaveWeights(const std::string &strFilename) const {
	::SaveWeights(NamedParameters(), strFilename);
}
