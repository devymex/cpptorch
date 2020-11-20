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
	for (const auto &pSubMod: children()) {
		NAMED_PARAMS namedParams;
		for (auto &params: pSubMod->named_parameters()) {
			namedParams[params.key()] = params.value();
		}
		if (!InitProc(pSubMod->name(), namedParams)) {
			for (const auto &param: namedParams) {
				LOG(INFO) << "Unintialized parameter: " << param.first;
			}
		}
	}
}

void BasicModel::LoadWeights(const std::string &strFilename) {
	auto loadedParams = ::LoadWeights(strFilename);
	for (const auto &param: named_parameters()) {
		auto iLoaded = loadedParams.find(param.key());
		if (iLoaded != loadedParams.end()) {
			param.value().set_data(iLoaded->second);
		} else {
			LOG(INFO) << "Unintialized parameter: " << param.key();
		}
	}
}

void BasicModel::SaveWeights(const std::string &strFilename) const {
	::SaveWeights(NamedParameters(), strFilename);
}

