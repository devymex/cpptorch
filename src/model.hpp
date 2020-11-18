#ifndef __MODEL_HPP
#define __MODEL_HPP

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn.h>

using NAMED_PARAMS = std::map<std::string, torch::Tensor>;
using WEIGHT_INIT_PROC = std::function<bool(const std::string&, NAMED_PARAMS&)>;

class NetImpl : public torch::nn::Module {
public:
	NetImpl();

	virtual void initialize(WEIGHT_INIT_PROC InitProc);

	virtual c10::IValue forward(std::vector<c10::IValue> inputs);
	
	virtual NAMED_PARAMS named_parameters() const;

	virtual void load_weights(const std::string &strFilename);

	virtual void save_weights(const std::string &strFilename) const;

protected:
	torch::nn::Linear m_Linear1 = nullptr;
	torch::nn::Linear m_Linear2 = nullptr;
};

class NetJITImpl : public NetImpl {
public:
	using JITMODULE = torch::jit::Module;
public:
	NetJITImpl() = default;
	NetJITImpl(JITMODULE &&jitMod);
	void train(bool on = true) override;

	c10::IValue forward(std::vector<c10::IValue> inputs) override;

	void initialize(WEIGHT_INIT_PROC InitProc) override;

	NAMED_PARAMS named_parameters() const override;

private:
	JITMODULE m_Module;
};

TORCH_MODULE(Net);
TORCH_MODULE(NetJIT);

#endif // #ifndef __MODEL_HPP