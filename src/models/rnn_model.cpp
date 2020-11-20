#include "../creator.hpp"
#include "../argman.hpp"
#include "basic_model.hpp"

class RNNModel : public BasicModel {
public:
	void Initialize(const nlohmann::json &jConf) override {
		ArgMan argMan;
		Arg<uint64_t> argCharsetSize("charset_size", 0, argMan);
		Arg<uint64_t> argHiddenSize("hidden_size", 0, argMan);
		Arg<uint64_t> argCategories("categories", 0, argMan);
		Arg<uint64_t> argNumLayers("num_layers", 1, argMan);
		ParseArgsFromJson(jConf, argMan);

		CHECK_GT(argCharsetSize(), 0);
		CHECK_GT(argHiddenSize(), 0);
		CHECK_GT(argCategories(), 0);
		CHECK_GT(argNumLayers(), 0);

		m_LSTM = register_module("LSTM", torch::nn::LSTM(torch::nn::LSTMOptions(
				argCharsetSize(), argHiddenSize()).num_layers(argNumLayers())));
		m_Softmax = register_module("softmax", torch::nn::Linear(
				argHiddenSize(), argCategories()));
	}

	torch::Tensor Forward(TENSOR_ARY inputs) override {
		auto input = inputs[0];
		CHECK_EQ(input.dim(), 3);
		int64_t nBatchSize = input.size(0);
		CHECK_GT(nBatchSize, 0);
		int64_t nStrLen = input.size(1);
		CHECK_GT(nStrLen, 0);

		auto memOpt = torch::TensorOptions(input.device()).dtype(input.dtype());
		std::vector<int64_t> shape = { m_LSTM->options.num_layers(),
				(int64_t)nBatchSize, m_LSTM->options.hidden_size() };
		auto h = torch::zeros(shape, memOpt);
		auto c = torch::zeros(shape, memOpt);
		input = input.transpose(0, 1);
		auto out = m_LSTM(input, std::make_tuple(std::move(h), std::move(c)));
		h = std::get<0>(out)[nStrLen - 1];
		auto softmax = torch::log_softmax(m_Softmax(h), 1);
		return softmax;
	}

private:
	torch::nn::LSTM m_LSTM = nullptr;
	torch::nn::Linear m_Softmax = nullptr;
};

REGISTER_CREATOR(BasicModel, RNNModel, "RNN");
