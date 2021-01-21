#include <algorithm>
#include <opencv2/opencv.hpp>
#include "../creator.hpp"
#include "basic_loss.hpp"

namespace tfunc = torch::nn::functional;

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
		auto nBatchSize = targets[0].size(0);
		CHECK_GT(nBatchSize, 0);
		auto truths = __ExtractTruthBoxes(targets[0]);

		// Get image size from meta information
		torch::Tensor tMeta = targets[1].cpu().contiguous();
		CHECK_EQ(tMeta.dim(), 2);
		CHECK_EQ(tMeta.size(0), nBatchSize);
		auto pMeta = (cv::Size2f*)tMeta.data_ptr<float>();
		std::vector<cv::Size2f> batchImgSize(pMeta, pMeta + nBatchSize);

		// Pre-process for predictions
		int64_t nNumVals = 0;
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
		__UpdateAnchorInfo(outputs);
		auto tPredVals = __ConcatenateOutputs(outputs); // batch*anc*row*col, val
		auto nNumBatchBoxes = tPredVals.size(0);

		// Calculate negative and positive delta(alpha and beta)
		m_Alpha.clear();
		m_Beta.clear();
		m_Alpha.resize(nNumBatchBoxes * nNumVals, 0.f);
		m_Beta.resize(nNumBatchBoxes * nNumVals, 0.f);
		__CalcNegativeDelta(tPredVals, truths);
		__CalcPositiveDelta(tPredVals, truths, batchImgSize);

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
	void __UpdateAnchorInfo(const TENSOR_ARY &outputs) {
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
			m_AncInfos[i].boxSize.width = m_AncSize[i].width;
			m_AncInfos[i].boxSize.height = m_AncSize[i].height;
			m_AncInfos[i].nOffset = m_nImgAncCnt;
			m_nImgAncCnt += m_AncInfos[i].cells.area();
		}
	}

	torch::Tensor __ConcatenateOutputs(const TENSOR_ARY &outputs) const {
		auto nBatchSize = outputs[0].sizes().front();
		auto nNumVals   = outputs[0].sizes().back();
		TENSOR_ARY flatOutputs;
		for (uint64_t i = 0; i < nBatchSize; ++i) {
			for (auto &out : outputs) {
				auto tFlat = out.slice(0, i, i + 1).reshape({-1, nNumVals});
				flatOutputs.emplace_back(std::move(tFlat));
			}
		}
		return torch::cat(std::move(flatOutputs), 0).cpu();
	}

	void __CalcNegativeDelta(const torch::Tensor &tPredVals,
			const std::vector<std::vector<TRUTH>> &truths) {
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
			const std::vector<std::vector<TRUTH>> &truths,
			const std::vector<cv::Size2f> &batchImgSize) {
		auto nBatchSize = truths.size();
		uint64_t nNumVals = tPredVals.sizes().back();
#ifdef DEBUG_DUMP_DELTA
		auto tDump = tPredVals.contiguous();
#endif
		for (uint64_t b = 0; b < nBatchSize; ++b) {
			for (auto &truth: truths[b]) {
				auto &truBox = truth.first;
				auto fScale = m_fIoUWeight * (2 - truBox.area());
				auto truCenter = (truBox.tl() + truBox.br()) / 2;

				auto bestAnc = __GetBestAnchor(truBox, batchImgSize[b]);
				auto ancInfo = bestAnc.first;
				auto iAnc = m_nImgAncCnt * b + bestAnc.second;
				auto ancSize = ancInfo.boxSize;
				ancSize.width /= batchImgSize[b].width;
				ancSize.height /= batchImgSize[b].height;

				auto pAlpha = &m_Alpha[iAnc * nNumVals];
				auto pBeta = &m_Beta[iAnc * nNumVals];
				pAlpha[0] = -fScale;
				pAlpha[1] = -fScale;
				pAlpha[2] = -fScale;
				pAlpha[3] = -fScale;
				pAlpha[8] = -1;
				pBeta[0]  = fScale * truCenter.x * ancInfo.cells.width;
				pBeta[1]  = fScale * truCenter.y * ancInfo.cells.height;
				pBeta[2]  = fScale * std::log(truBox.width / ancSize.width);
				pBeta[3]  = fScale * std::log(truBox.height / ancSize.height);
				pBeta[8]  = 1;

				if (pAlpha[m_nNumBoxVals + truth.second] == 0) {
					for (uint64_t i = 0; i < nNumVals - m_nNumBoxVals; ++i) {
						pAlpha[m_nNumBoxVals + i] = -1;
					}
				}
				pAlpha[m_nNumBoxVals + truth.second] = -1;
				pBeta[m_nNumBoxVals + truth.second] = 1;

				static int n = 0;
#ifdef DEBUG_DUMP_DELTA
				std::ostringstream oss;
				auto p = tDump.data_ptr<float>() + iAnc * nNumVals;
				oss << n++ << " " << p[8];
				for (int64_t j = 0; j < 4; ++j) {
					oss << " (" << -pAlpha[j] * p[j] << "," << pBeta[j] << ")";
				}
				LOG(INFO) << oss.str();
#endif
			}
		}
	}

	std::pair<ANCHOR_INFO, int64_t> __GetBestAnchor(
			const cv::Rect2f &truBox, const cv::Size2f &imgSize) {
		cv::Point2f truCenter = (truBox.tl() + truBox.br()) / 2;
		float fBestIoU = 0.f;
		uint64_t iBestAnchor = 0;
		for (uint64_t i = 0; i < m_AncInfos.size(); ++i) {
			auto ancSize = m_AncInfos[i].boxSize;
			ancSize.width /= imgSize.width;
			ancSize.height /= imgSize.height;
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
