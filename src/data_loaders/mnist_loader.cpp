#include <glog/logging.h>

#include "../argman.hpp"
#include "mnist_loader.hpp"

using namespace torch::data::datasets;

MNISTLoader::MNISTLoader(const nlohmann::json &jConf) {
	Initialize(jConf);
}

void MNISTLoader::Initialize(const nlohmann::json &jConf) {
	BatchLoader::Initialize(jConf);

	ArgMan argMan;
	Arg<std::string> argDataRoot("data_root", argMan);
	Arg<bool> argTrainSet("train_set", argMan);
	ParseArgsFromJson(jConf, argMan);

	m_pMNIST.reset(new MNIST(argDataRoot(), argTrainSet() ?
			MNIST::Mode::kTrain : MNIST::Mode::kTest));
}

size_t MNISTLoader::Size() const {
	CHECK(m_pMNIST->size().has_value());
	return m_pMNIST->size().value();
}

void MNISTLoader::_LoadBatch(std::vector<size_t> indices, torch::Device device,
		torch::Tensor &tData, torch::Tensor &tLabel) {
	CHECK(!indices.empty());
	std::vector<torch::Tensor> dataAry;
	std::vector<torch::Tensor> targetAry;
	for (size_t i = 0; i < indices.size(); ++i) {
		CHECK_LT(indices[i], Size());
		auto sample = m_pMNIST->get(indices[i]);
		if (sample.target.dim() == 0) {
			sample.target = sample.target.reshape({1});
		}
		dataAry.emplace_back(sample.data);
		targetAry.emplace_back(sample.target);
	}
	tData = torch::cat(dataAry).to(device);
	tLabel = torch::cat(targetAry).to(device);
}
