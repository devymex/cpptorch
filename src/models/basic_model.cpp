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

void BasicModel::InitWeights(WEIGHT_INIT_PROC InitProc) {
	for (auto &submod : modules()) {
		NAMED_PARAMS namedParams;
		for (auto &params: submod->named_parameters()) {
			namedParams[params.key()] = params.value();
		}
		if (!InitProc(submod->name(), namedParams)) {
			for (const auto &param: namedParams) {
				LOG(INFO) << "Unintialized parameter: " << param.first;
			}
		}
	}
}

void BasicModel::LoadWeights(const std::string &strFilename) {
	auto loadedParams = ::LoadWeights(strFilename);
	for (auto &param: NamedParameters()) {
		auto iLoaded = loadedParams.find(param.first);
		if (iLoaded != loadedParams.end()) {
			param.second.set_data(iLoaded->second);
		} else {
			LOG(INFO) << "Unintialized parameter: " << param.first;
		}
	}
}

void BasicModel::SaveWeights(const std::string &strFilename) const {
	::SaveWeights(NamedParameters(), strFilename);
}

