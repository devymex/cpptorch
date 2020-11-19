#include "../creator.hpp"
#include "../argman.hpp"
#include "basic_model.hpp"

class RNNModel : public BasicModel {
public:
	void Initialize(const nlohmann::json &jConf) override {
		ArgMan argMan;
		Arg<uint32_t> argCharsetSize("charset_size", argMan);
		Arg<uint32_t> argCategories("categories", argMan);
		ParseArgsFromJson(jConf, argMan);

		m_i2h = register_module("i2h", torch::nn::Linear(
				argCharsetSize() * 2, argCharsetSize()));
		m_i2o = register_module("i2o", torch::nn::Linear(
				argCharsetSize() * 2, argCategories()));
	}

	torch::Tensor Forward(std::vector<torch::Tensor> inputs) override {
		auto input = inputs[0];
		CHECK_EQ(input.dim(), 3);
		size_t nStrLen = input.size(1);
		input = input.transpose(0, 1);
		auto hidden = torch::zeros_like(input[0]);
		for (size_t i = 0; i < nStrLen; ++i) {
			hidden = torch::cat({input[i], hidden}, 1);
			if (i < nStrLen - 1) {
				hidden = m_i2h(hidden);
			}
		}
		auto output = torch::log_softmax(m_i2o(hidden), 1);
		return output;
	}

protected:
	torch::nn::Linear m_i2h = nullptr;
	torch::nn::Linear m_i2o = nullptr;
};

REGISTER_CREATOR(BasicModel, RNNModel, "RNN");
