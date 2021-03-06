#include <memory>
#include <glog/logging.h>
#include <torch/data/datasets/mnist.h>
#include "../json.hpp"
#include "../creator.hpp"
#include "../argman.hpp"
#include "batch_loader.hpp"

using namespace torch::data::datasets;

class MNISTLoader : public BatchLoader {
public:
	void Initialize(const nlohmann::json &jConf) override {
		BatchLoader::Initialize(jConf);

		ArgMan argMan;
		Arg<std::string> argDataRoot("data_root", argMan);
		Arg<bool> argTrainSet("train_set", argMan);
		ParseArgsFromJson(jConf, argMan);

		m_pMNIST.reset(new MNIST(argDataRoot(), argTrainSet() ?
				MNIST::Mode::kTrain : MNIST::Mode::kTest));
	}
	uint64_t Size() const override {
		CHECK(m_pMNIST->size().has_value());
		return m_pMNIST->size().value();
	}

protected:
	void _LoadBatch(std::vector<uint64_t> indices,
			TENSOR_ARY &data, TENSOR_ARY &targets) override {
		CHECK(!indices.empty());
		TENSOR_ARY dataAry;
		TENSOR_ARY targetAry;
		for (uint64_t i = 0; i < indices.size(); ++i) {
			CHECK_LT(indices[i], Size());
			auto sample = m_pMNIST->get(indices[i]);
			if (sample.data.dim() == 3) {
				auto shape = sample.data.sizes().vec();
				shape.insert(shape.begin(), 1);
				sample.data = sample.data.reshape(shape);
			}
			if (sample.target.dim() == 0) {
				sample.target = sample.target.reshape({1});
			}
			dataAry.emplace_back(std::move(sample.data));
			targetAry.emplace_back(std::move(sample.target));
		}
		data = TENSOR_ARY { torch::cat(dataAry) };
		targets = TENSOR_ARY { torch::cat(targetAry) };
	}

protected:
	std::unique_ptr<torch::data::datasets::MNIST> m_pMNIST;
};

REGISTER_CREATOR(BatchLoader, MNISTLoader, "MNIST");
