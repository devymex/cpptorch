#ifndef __MNIST_LOADER_HPP
#define __MNIST_LOADER_HPP

#include <memory>
#include <torch/data/datasets/mnist.h>
#include "../json.hpp"
#include "batch_loader.hpp"

class MNISTLoader : public BatchLoader {
public:
	MNISTLoader(const nlohmann::json &jConf);
	virtual void Initialize(const nlohmann::json &jConf);
	size_t Size() const override;

protected:
	void _LoadBatch(std::vector<size_t> indices, torch::Device device,
			torch::Tensor &tData, torch::Tensor &tLabel) override;

protected:
	std::unique_ptr<torch::data::datasets::MNIST> m_pMNIST;
};

#endif // #ifndef __MNIST_LOADER_HPP