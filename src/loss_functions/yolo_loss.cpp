#include <algorithm>
#include <opencv2/opencv.hpp>
#include "../creator.hpp"
#include "basic_loss.hpp"

namespace tfunc = torch::nn::functional;

struct BBOX {
	BBOX(): p1(0, 0), p2(0, 0), c(-1) {
	}
	BBOX(cv::Point2f _p1, cv::Point2f _p2): p1(_p1), p2(_p2), c(-1) {
	}
	BBOX(cv::Point2f _p1, cv::Point2f _p2, float _c): p1(_p1), p2(_p2), c(_c) {
	}
	BBOX(const BBOX &other): p1(other.p1), p2(other.p2), c(other.c) {
	}
	float area() const {
		return (p2.x - p1.x) * (p2.y - p1.y);
	}
	void fix() {
		if (p1.x > p2.x) {
			std::swap(p1.x, p2.x);
		}
		if (p1.y > p2.y) {
			std::swap(p1.y, p2.y);
		}
	}
	cv::Point2f p1;
	cv::Point2f p2;
	float c;
};

BBOX operator & (BBOX b1, BBOX b2) {
	b1.fix();
	b2.fix();
	BBOX res({ std::max(b1.p1.x, b2.p1.x), std::max(b1.p1.y, b2.p1.y) },
			 { std::min(b1.p2.x, b2.p2.x), std::min(b1.p2.y, b2.p2.y) });
	res.c = b1.c == b2.c ? b1.c : -1;
	return res;
}

float IoU(const cv::Rect2f b1, const cv::Rect2f &b2) {
	float fIOU = (b1 & b2).area();
	if (fIOU > 0) {
		fIOU /= b1.area() + b2.area() - fIOU;
	}
	return fIOU;
}

struct ANCHOR_INFO {
	cv::Size cells;
	cv::Size2f boxSize;
	uint64_t nOffset;
};

class YOLOLoss : public BasicLoss {
	using TRUTH = std::pair<cv::Rect2f, int32_t>;
	const uint64_t m_nNumBoxVals = 9;
	const float m_fNegProbThres = 0.25f;
	const float m_fNegIgnoreIoU = 0.7f;
	const float m_fIoUWeight = 0.75f;
	const std::vector<cv::Size2f> m_AncSize = {
		cv::Size2f(116,90), cv::Size2f(156,198), cv::Size2f(373,326),
		cv::Size2f(30,61),  cv::Size2f(62,45),   cv::Size2f(59,119),
		cv::Size2f(10,13),  cv::Size2f(16,30),   cv::Size2f(33,23)
	};

	std::vector<ANCHOR_INFO> m_AncInfos;
	uint64_t m_nImgAncCnt = 0; // amount of different size anchors in an image
	std::vector<float> m_Alpha;
	std::vector<float> m_Beta;

public:
	float Backward(TENSOR_ARY outputs, TENSOR_ARY targets) override {
		auto tLoss = __CalcLoss(outputs, targets);
		float fLoss = tLoss.item<float>();
		tLoss.backward();
		return fLoss;
	}

	float Evaluate(TENSOR_ARY outputs, TENSOR_ARY targets) override {
		auto tLoss = __CalcLoss(outputs, targets);
		float fLoss = tLoss.item<float>();
		return fLoss;
	}

	std::string FlushResults() {
		return "";
	}

private:
	torch::Tensor __CalcLoss(const TENSOR_ARY &outputs,
			const TENSOR_ARY &targets) {
		// Preprocess for truth-boxes
		CHECK_EQ(targets.size(), 2);
		uint64_t nBatchSize = targets[0].size(0);
		CHECK_GT(nBatchSize, 0);
		torch::Tensor tMeta = targets[1].cpu();
		cv::Size2f imgSize = {tMeta[0].item<float>(), tMeta[1].item<float>()};
		auto truths = __ExtractTruthBoxes(targets[0]);

		// Pre-process for predictions
		uint64_t nNumVals = 0;
		CHECK(!outputs.empty());
		for (auto &out: outputs) {
			CHECK_EQ(out.dim(), 5); // batch, anc, rows, cols, vals
			CHECK_EQ(out.size(0), nBatchSize);
			if (nNumVals == 0) {
				nNumVals = out.size(4);
			} else {
				CHECK_EQ(nNumVals, out.size(4));
			}
		}
		__UpdateAnchorInfo(outputs, imgSize);
		auto tPredVals = __ConcatenateOutputs(outputs);
		auto nNumBatchBoxes = tPredVals.size(0);

		// Calculate negative and positive delta(alpha and beta)
		m_Alpha.clear();
		m_Beta.clear();
		m_Alpha.resize(nNumBatchBoxes * nNumVals, 0.f);
		m_Beta.resize(nNumBatchBoxes * nNumVals, 0.f);
		__CalcNegativeDelta(tPredVals, truths);
		__CalcPositiveDelta(tPredVals, truths);

		// Computing loss and backward
		std::vector<long> deltaShape = {(long)nNumBatchBoxes, (long)nNumVals};
		auto optNoGrad = torch::TensorOptions(c10::requires_grad(false));
		auto tAlpha = torch::from_blob(m_Alpha.data(), deltaShape, optNoGrad);
		auto tBeta = torch::from_blob(m_Beta.data(), deltaShape, optNoGrad);
		return torch::square(tPredVals * tAlpha + tBeta).sum() / (long)nNumVals;
	}

	std::vector<std::vector<TRUTH>> __ExtractTruthBoxes(const torch::Tensor &tTruth) {
		torch::Tensor tTruthCPU = tTruth.cpu().contiguous();
		uint64_t nBatchSize = tTruthCPU.size(0);
		std::vector<std::vector<TRUTH>> truths(nBatchSize);
		for (uint64_t b = 0; b < nBatchSize; ++b) {
			auto tBoxes = tTruthCPU[b];
			auto pBoxes = tBoxes.data_ptr<float>();
			for (uint i = 0; i < tBoxes.size(0); ++i) {
				// Truth-Box: [cls_id, x1, y1, x2, y2]
				int32_t nClsId = pBoxes[i * 5];
				if (nClsId < 0) { // unavailable truth-boxes
					break; // cls_id < 0: last turth-box of image
				}
				// after cls_id are x1y1(Top-Left) and x2y2(Bottom-Right).
				auto pTLBR = (cv::Point2f*)(pBoxes + i * 5 + 1);
				truths[b].emplace_back(cv::Rect2f {pTLBR[0], pTLBR[1]}, nClsId);
			}
		}
		return std::move(truths);
	}

	// To support multiple yolo layers with different number of anchors
	void __UpdateAnchorInfo(const TENSOR_ARY &outputs, cv::Size imgSize) {
		std::vector<cv::Size> ancGrids;
		for (auto &out : outputs) {
			// Get num_cols and num_rows from shape of outputs
			for (uint64_t i = 0; i < out.size(1); ++i) {
				ancGrids.emplace_back(out.size(3), out.size(2));
			}
		} // But the total number should be matched with the given anchors
		CHECK_EQ(ancGrids.size(), m_AncSize.size());

		m_AncInfos.resize(m_AncSize.size());
		m_nImgAncCnt = 0;
		for (uint64_t i = 0; i < m_AncInfos.size(); ++i) {
			m_AncInfos[i].cells.width = ancGrids[i].width;
			m_AncInfos[i].cells.height = ancGrids[i].height;
			m_AncInfos[i].boxSize.width = m_AncSize[i].width / imgSize.width;
			m_AncInfos[i].boxSize.height = m_AncSize[i].height / imgSize.height;
			m_AncInfos[i].nOffset = m_nImgAncCnt;
			m_nImgAncCnt += m_AncInfos[i].cells.area();
		}
	}

	torch::Tensor __ConcatenateOutputs(const TENSOR_ARY &outputs) {
		TENSOR_ARY flatOutputs;
		for (auto &out : outputs) {
			flatOutputs.emplace_back(out.view({-1, out.sizes().back()}));
		}
		return torch::cat(std::move(flatOutputs), 0).cpu();
	}

	void __CalcNegativeDelta(const torch::Tensor &tPredVals,
			std::vector<std::vector<TRUTH>> &truths) {
		auto nNumBatchPredBoxes = tPredVals.size(0);
		CHECK_EQ(nNumBatchPredBoxes % truths.size(), 0);
		auto nNumImagePredBoxes = nNumBatchPredBoxes/ truths.size();

		// set mask=1 for all negative boxes whose maximum probs greater than
		//    m_fNegProbThres
		auto tPredProbs = tPredVals.slice(1, m_nNumBoxVals);
		auto tNegMask = torch::gt(torch::amax(tPredProbs, 1), m_fNegProbThres);
		tNegMask = tNegMask.to(torch::kInt32).contiguous();

		auto tPredBoxes = tPredVals.slice(1, 4, 8).contiguous();
		auto pPredBoxes = tPredBoxes.data_ptr<float>(); // [x1, y1, x2, y2]
		auto pNegMask = tNegMask.data_ptr<int32_t>();
		for (uint64_t i = 0; i < nNumBatchPredBoxes; ++i) {
			if (pNegMask[i]) {
				auto pTLBR = (cv::Point2f*)(pPredBoxes + i * 4);
				cv::Rect2f predBox = { pTLBR[0], pTLBR[1] };
				auto fBestIOU = 0.f;
				for (auto &truth : truths[i / nNumImagePredBoxes]) {
					fBestIOU = std::max(fBestIOU, IoU(predBox, truth.first));
				}
				if (fBestIOU <= m_fNegIgnoreIoU) {
					m_Alpha[i * tPredVals.sizes().back() + 8] = -1;
				}
			}
		}
	}
	void __CalcPositiveDelta(const torch::Tensor &tPredVals,
			std::vector<std::vector<TRUTH>> &truths) {
		auto nBatchSize = truths.size();
		uint64_t nNumVals = tPredVals.sizes().back();
		for (uint64_t b = 0; b < nBatchSize; ++b) {
			for (auto &truth: truths[b]) {
				auto &truBox = truth.first;
				auto fScale = m_fIoUWeight * (2 - truBox.area());
				auto truCenter = (truBox.tl() + truBox.br()) / 2;

				auto bestAnc = __GetBestAnchor(truBox);
				auto ancInfo = bestAnc.first;
				auto iAnc = m_nImgAncCnt * b + bestAnc.second;
				auto pAlpha = &m_Alpha[iAnc * nNumVals];
				auto pBeta = &m_Beta[iAnc * nNumVals];

				pAlpha[0] = -fScale;
				pAlpha[1] = -fScale;
				pAlpha[2] = -fScale;
				pAlpha[3] = -fScale;
				pAlpha[8] = -1;
				pBeta[0]  = fScale * truCenter.x * ancInfo.cells.width;
				pBeta[1]  = fScale * truCenter.y * ancInfo.cells.height;
				pBeta[2]  = fScale * truBox.width / ancInfo.boxSize.width;
				pBeta[3]  = fScale * truBox.height / ancInfo.boxSize.height;
				pBeta[8]  = 1;

				if (pAlpha[m_nNumBoxVals + truth.second] == 0) {
					for (uint64_t i = 0; i < nNumVals - m_nNumBoxVals; ++i) {
						pAlpha[m_nNumBoxVals + i] = -1;
					}
				}
				pAlpha[m_nNumBoxVals + truth.second] = -1;
				pBeta[m_nNumBoxVals + truth.second] = 1;
			}
		}
	}

	std::pair<ANCHOR_INFO, int64_t> __GetBestAnchor(
			const cv::Rect2f &truBox) {
		cv::Point2f truCenter = (truBox.tl() + truBox.br()) / 2;
		float fBestIoU = 0.f;
		uint64_t iBestAnchor = 0;
		for (uint64_t i = 0; i < m_AncInfos.size(); ++i) {
			const auto &ancSize = m_AncInfos[i].boxSize;
			// to align the center of the anchor box to center of the truth
			cv::Point2f ancTL = truCenter - cv::Point2f(ancSize) / 2;
			float fIoU = IoU(truBox, cv::Rect2f(ancTL, ancSize));
			if (fIoU > fBestIoU) {
				fBestIoU = fIoU;
				iBestAnchor = i;
			}
		}
		const auto &bestAnc = m_AncInfos[iBestAnchor];
		auto c = uint64_t(truCenter.x * bestAnc.cells.width);
		auto r = uint64_t(truCenter.y * bestAnc.cells.height);
		auto iBox = r * bestAnc.cells.width + c;
		CHECK_LT(iBox, bestAnc.cells.area());
		return std::make_pair(bestAnc, bestAnc.nOffset + iBox);
	}
};

REGISTER_CREATOR(BasicLoss, YOLOLoss, "YOLO");
