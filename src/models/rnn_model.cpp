#include "../creator.hpp"
#include "../argman.hpp"
#include "basic_model.hpp"

class RNNModel : public BasicModel {
public:
	void Initialize(const nlohmann::json &jConf) override {
		ArgMan argMan;
		Arg<uint64_t> argCharsetSize("charset_size", argMan);
		Arg<uint64_t> argCategories("categories", argMan);
		Arg<uint64_t> argHiddenSize("hidden_size", 0, argMan);
		Arg<uint64_t> argStrLen("string_length", 0, argMan);
		ParseArgsFromJson(jConf, argMan);

		m_nHiddenSize = argHiddenSize();
		CHECK_GT(m_nHiddenSize, 0);
		m_nStrLen = argStrLen();
		CHECK_GT(m_nStrLen, 0);
		auto nISize = argCharsetSize() + m_nHiddenSize;
		auto nOSize = m_nHiddenSize;
		m_gateF = register_module("gateF", torch::nn::Linear(nISize, nOSize));
		m_gateI = register_module("gateI", torch::nn::Linear(nISize, nOSize));
		m_gateC = register_module("gateC", torch::nn::Linear(nISize, nOSize));
		m_gateO = register_module("gateO", torch::nn::Linear(nISize, nOSize));
		m_Softmax = register_module("softmax",
				torch::nn::Linear(nOSize, argCategories()));
		torch::nn::LSTM
	}

	torch::Tensor Forward(TENSOR_ARY inputs) override {
		CHECK_GT(m_nHiddenSize, 0);
		auto input = inputs[0];
		CHECK_EQ(input.dim(), 3);
		int64_t nBatchSize = input.size(0);
		CHECK_GT(nBatchSize, 0);
		int64_t nStrLen = input.size(1);
		CHECK_GT(nStrLen, 0);
		CHECK_EQ(nStrLen, m_nStrLen);
		input = input.transpose(0, 1);
		auto memOpt = torch::TensorOptions(input.device()).dtype(input.dtype());
		auto c = torch::zeros({nBatchSize, (int64_t)m_nHiddenSize}, memOpt);
		auto h = torch::zeros({nBatchSize, (int64_t)m_nHiddenSize}, memOpt);
		for (uint64_t i = 0; i < nStrLen; ++i) {
			__LSTMCell(h, c, input[i]);
		}
		auto softmax = torch::log_softmax(m_Softmax(h), 1);
		return softmax;
	}

	PARAM_OPTION GetParamOption(const std::string &strParamName) const override {
		if (strParamName.find("gate") != std::string::npos) {
			return PARAM_OPTION { 1.f / m_nStrLen, 1.f / m_nStrLen };
		}
		return BasicModel::GetParamOption(strParamName);
	}

private:
	void __LSTMCell(torch::Tensor &h, torch::Tensor &c, torch::Tensor x) {
		auto combine = torch::cat({h, x}, 1);
		auto f = torch::sigmoid(m_gateF(combine));
		auto i = torch::sigmoid(m_gateI(combine));
		auto o = torch::sigmoid(m_gateO(combine));
		c = f.mul(c) + i.mul(torch::tanh(m_gateC(combine)));
		h = o.mul(torch::tanh(c));
	}

private:
	uint64_t m_nHiddenSize = 0;
	uint64_t m_nStrLen = 0;
	torch::nn::Linear m_gateF = nullptr;
	torch::nn::Linear m_gateI = nullptr;
	torch::nn::Linear m_gateC = nullptr;
	torch::nn::Linear m_gateO = nullptr;
	torch::nn::Linear m_Softmax = nullptr;
};

REGISTER_CREATOR(BasicModel, RNNModel, "RNN");
