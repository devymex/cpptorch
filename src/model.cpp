#include <fstream>
#include "model.hpp"
#include "weights.hpp"
#include "torch/data/datasets/mnist.h"

NetImpl::NetImpl() {
	m_Linear1 = register_module("Linear1", torch::nn::Linear(28 * 28, 128));
	m_Linear2 = register_module("Linear2", torch::nn::Linear(128, 10));
}

c10::IValue NetImpl::forward(std::vector<c10::IValue> inputs) {
	auto x = inputs[0].toTensor().flatten(1);
	x = torch::relu(m_Linear1(x));
	x = torch::relu(m_Linear2(x));
	return x;
}

void NetImpl::initialize(WEIGHT_INIT_PROC InitProc) {
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

NAMED_PARAMS NetImpl::named_parameters() const {
	NAMED_PARAMS namedParams;
	for (const auto &params : torch::nn::Module::named_parameters()) {
		namedParams[params.key()] = params.value();
	}
	return namedParams;
}

void NetImpl::load_weights(const std::string &strFilename) {
	auto loadedParams = LoadWeights(strFilename);
	for (auto &param: named_parameters()) {
		auto iLoaded = loadedParams.find(param.first);
		if (iLoaded != loadedParams.end()) {
			param.second.set_data(iLoaded->second);
		} else {
			LOG(INFO) << "Unintialized parameter: " << param.first;
		}
	}
}

void NetImpl::save_weights(const std::string &strFilename) const {
	SaveWeights(named_parameters(), strFilename);
}


NetJITImpl::NetJITImpl(JITMODULE &&jitMod)
	: m_Module(std::forward<JITMODULE>(jitMod)) {
}

void NetJITImpl::train(bool on) {
	m_Module.train(on);
}

c10::IValue NetJITImpl::forward(std::vector<c10::IValue> inputs) {
	return m_Module.forward(std::move(inputs));
}

void NetJITImpl::initialize(WEIGHT_INIT_PROC InitProc) {
	for (const auto &submod : m_Module.modules()) {
		auto typeName = submod.type().get()->name();
		if (typeName.has_value() && submod.modules().size() == 1) {
			std::string strTypeName = typeName.value().name();
			NAMED_PARAMS namedParams;
			for (const auto &params: submod.named_parameters()) {
				namedParams[params.name] = params.value;
			}
			InitProc(strTypeName, namedParams);
		}
	}
}

NAMED_PARAMS NetJITImpl::named_parameters() const {
	NAMED_PARAMS namedParams;
	for (const auto &params : m_Module.named_parameters()) {
		namedParams[params.name] = params.value;
	}
	return namedParams;
}
