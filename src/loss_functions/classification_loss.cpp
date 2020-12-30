#include "../creator.hpp"
#include "basic_loss.hpp"

namespace tfunc = torch::nn::functional;

class ClassificationLoss : public BasicLoss {
public:
	float Backward(torch::Tensor tOutput, torch::Tensor tTarget) override {
		uint64_t nBatchSize = tOutput.size(0);
		CHECK_GT(nBatchSize, 0);
		torch::Tensor tLoss = tfunc::nll_loss(tOutput, tTarget);
		float fLoss = tLoss.item().toFloat() * nBatchSize;
		tLoss.backward();
		return fLoss;
	}

	float Evaluate(torch::Tensor tOutput, torch::Tensor tTarget) override {
		uint64_t nBatchSize = tOutput.size(0);
		uint64_t nClassNum = tOutput.size(1);
		torch::Tensor tLoss = tfunc::nll_loss(tOutput, tTarget);
		float fLoss = tLoss.item().toFloat() * nBatchSize;

		auto tSum = tOutput.argmax(1).eq(tTarget).sum();
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
