#include "../creator.hpp"
#include "basic_loss.hpp"

namespace tfunc = torch::nn::functional;

class ClassificationLoss : public BasicLoss {
public:
	float Backward(TENSOR_ARY outputs, TENSOR_ARY targets) override {
		CHECK_EQ(outputs.size(), 1);
		auto &tOutput = outputs[0];
		uint64_t nBatchSize = tOutput.size(0);
		CHECK_GT(nBatchSize, 0);
		torch::Tensor tLoss = tfunc::nll_loss(tOutput, targets[0]);
		float fLoss = tLoss.item().toFloat() * nBatchSize;
		tLoss.backward();
		return fLoss;
	}

	float Evaluate(TENSOR_ARY outputs, TENSOR_ARY target) override {
		CHECK_EQ(outputs.size(), 1);
		auto &tOutput = outputs[0];
		uint64_t nBatchSize = tOutput.size(0);
		uint64_t nClassNum = tOutput.size(1);
		torch::Tensor tLoss = tfunc::nll_loss(tOutput, target[0]);
		float fLoss = tLoss.item().toFloat() * nBatchSize;

		auto tSum = tOutput.argmax(1).eq(target[0]).sum();
		m_nCorrectCnt += tSum.item().toLong();
		m_nTotalCnt += nBatchSize;

		auto tProb = tOutput.to(torch::kCPU);
		for (uint64_t i = 0; i < nBatchSize; ++i) {
			float *pSoftmax = tProb[i].data_ptr<float>();
			m_Results.insert(m_Results.end(), pSoftmax, pSoftmax + nClassNum);
		}
		return fLoss;
	}

	std::string FlushResults() {
		float fAcc = (float)m_nCorrectCnt / m_nTotalCnt;
		m_nCorrectCnt = 0;
		m_nTotalCnt = 0;
		m_Results.clear();
		return "acc=" + std::to_string(fAcc);
	}

private:
	int64_t m_nCorrectCnt = 0;
	int64_t m_nTotalCnt = 0;
	std::vector<float> m_Results;
};

REGISTER_CREATOR(BasicLoss, ClassificationLoss, "Classification");
