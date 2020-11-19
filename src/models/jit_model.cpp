
#include "basic_model.hpp"

// class NetJITImpl : public NetImpl {
// public:
// 	using JITMODULE = torch::jit::Module;
// public:
// 	NetJITImpl() = default;
// 	NetJITImpl(JITMODULE &&jitMod);
// 	void train(bool on = true) override;

// 	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs) override;

// 	void initialize(WEIGHT_INIT_PROC InitProc) override;

// 	NAMED_PARAMS named_parameters() const override;

// private:
// 	JITMODULE m_Module;
// };

// TORCH_MODULE(Net);
// TORCH_MODULE(NetJIT);

// NetJITImpl::NetJITImpl(JITMODULE &&jitMod)
// 	: m_Module(std::forward<JITMODULE>(jitMod)) {
// }

// void NetJITImpl::train(bool on) {
// 	m_Module.train(on);
// }

// torch::Tensor NetJITImpl::forward(std::vector<torch::Tensor> inputs) {
// 	std::vector<c10::IValue> _inputs;
// 	for (auto t : inputs) {
// 		_inputs.push_back(t);
// 	}
// 	return m_Module.forward(std::move(_inputs)).toTensor();
// }

// void NetJITImpl::initialize(WEIGHT_INIT_PROC InitProc) {
// 	for (const auto &submod : m_Module.modules()) {
// 		auto typeName = submod.type().get()->name();
// 		if (typeName.has_value() && submod.modules().size() == 1) {
// 			std::string strTypeName = typeName.value().name();
// 			NAMED_PARAMS namedParams;
// 			for (const auto &params: submod.named_parameters()) {
// 				namedParams[params.name] = params.value;
// 			}
// 			InitProc(strTypeName, namedParams);
// 		}
// 	}
// }

// NAMED_PARAMS NetJITImpl::named_parameters() const {
// 	NAMED_PARAMS namedParams;
// 	for (const auto &params : m_Module.named_parameters()) {
// 		namedParams[params.name] = params.value;
// 	}
// 	return namedParams;
// }
