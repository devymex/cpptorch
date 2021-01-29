#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>
#include "../creator.hpp"
#include "../argman.hpp"
#include "../ctimer.hpp"
#include "basic_loss.hpp"

namespace tfunc = torch::nn::functional;

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
	const std::vector<cv::Size2f> m_AncSize = {
		cv::Size2f(116,90), cv::Size2f(156,198), cv::Size2f(373,326),
		cv::Size2f(30,61),  cv::Size2f(62,45),   cv::Size2f(59,119),
		cv::Size2f(10,13),  cv::Size2f(16,30),   cv::Size2f(33,23)
	};
	float m_fXYWeight = 1.f;
	float m_fSizeWeight = 1.f;
	float m_fConfWeight = 1.f;
	float m_fProbWeight = 1.f;

	std::vector<ANCHOR_INFO> m_AncInfos;
	uint64_t m_nImgAncCnt = 0; // amount of different size anchors in an image

	torch::nn::BCELoss m_BCELoss;
	torch::nn::MSELoss m_MSELoss;

public:
	void Initialize(const nlohmann::json &jConf) override {
		BasicLoss::Initialize(jConf);
		ArgMan argMan;
		Arg<float> argXYWeight("xy_weight", 1.0f, argMan);
		Arg<float> argConfWeight("conf_weight", 1.0f, argMan);
		Arg<float> argSizeWeight("size_weight", 1.0f, argMan);
		Arg<float> argProbWeight("prob_weight", 1.0f, argMan);
		ParseArgsFromJson(jConf, argMan);

		m_fXYWeight = argXYWeight();
		m_fConfWeight = argConfWeight();
		m_fSizeWeight = argSizeWeight();
		m_fProbWeight = argProbWeight();

		m_BCELoss->options.reduction() = torch::kNone;
		m_MSELoss->options.reduction() = torch::kNone;
	}

	float Backward(TENSOR_ARY outputs, TENSOR_ARY targets) override {
		auto [tLoss, nNumAncs] = __CalcLoss(outputs, targets);
		(tLoss / nNumAncs).backward();
		return tLoss.item<float>();
	}

	float Evaluate(TENSOR_ARY outputs, TENSOR_ARY targets) override {
		auto [tLoss, nNumAncs] = __CalcLoss(outputs, targets);
		return tLoss.item<float>();
	}

	std::string FlushResults() {
		return "";
	}

private:
	std::pair<torch::Tensor, int64_t> __CalcLoss(const TENSOR_ARY &outputs,
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
		auto pMeta = (cv::Size*)tMeta.data_ptr<int32_t>();
		std::vector<cv::Size> batchImgSize(pMeta, pMeta + nBatchSize);

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
		auto tNegMask = __GetNegMask(tPredVals, truths);
		auto tLoss = __CalcLoss(tPredVals, tNegMask, truths, batchImgSize);
		return std::make_pair(tLoss, tPredVals.size(0));
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

	torch::Tensor __GetNegMask(const torch::Tensor &tPredVals,
			const std::vector<std::vector<TRUTH>> &truths) {
		auto nBatchSize = truths.size();
		auto nNumBatchAncs = tPredVals.size(0);
		auto nNumImageAncs = nNumBatchAncs / nBatchSize;
		auto nNumVals = tPredVals.size(1);
		CHECK_EQ(nNumBatchAncs % nBatchSize, 0);

		auto tPredBoxes = tPredVals.slice(1, 4, 8).contiguous();
		auto tPredConf = tPredVals.slice(1, 8, 9);
		auto tPredProbs = tPredVals.slice(1, 9, nNumVals).detach();
		auto pPredBoxes = tPredBoxes.data_ptr<float>(); // [x1, y1, x2, y2]

		// set mask=1 for all negative boxes whose maximum probs greater than
		//    m_fNegProbThres
		auto tNegMask = torch::gt(torch::amax(tPredProbs, 1), m_fNegProbThres);
		tNegMask = tNegMask.requires_grad_(false).to(torch::kInt32).contiguous();
		auto pNegMask = tNegMask.data_ptr<int32_t>();
		for (uint64_t i = 0; i < nNumBatchAncs; ++i) {
			if (pNegMask[i]) {
				auto pTLBR = (cv::Point2f*)(pPredBoxes + i * 4);
				cv::Rect2f predBox = { pTLBR[0], pTLBR[1] };
				auto fBestIOU = 0.f;
				for (auto &truth : truths[i / nNumImageAncs]) {
					fBestIOU = std::max(fBestIOU, IoU(predBox, truth.first));
				}
				if (fBestIOU > m_fNegIgnoreIoU) {
					pNegMask[i] = 0;
				}
			}
		}
		return tNegMask;
	}

	torch::Tensor __CalcLoss(torch::Tensor &tPredVals, torch::Tensor &tNegMask,
			const std::vector<std::vector<TRUTH>> &truths,
			const std::vector<cv::Size> &batchImgSize) {
		auto nBatchSize = truths.size();
		auto nNumBatchAncs = tPredVals.size(0);
		auto nNumVals = tPredVals.sizes().back();
		auto nNumClasses = nNumVals - 9;

		std::vector<int64_t> posIndices;
		std::vector<float> xyTruth, xyScale, whTruth, whScale, probsTruth;
		auto pNegMask = tNegMask.data_ptr<int32_t>();
		for (uint64_t b = 0; b < nBatchSize; ++b) {
			for (auto &truth: truths[b]) {
				auto &truBox = truth.first;

				auto [ancInfo, ancPos] = __GetBestAnchor(truBox, batchImgSize[b]);
				auto iAncInBatch = m_nImgAncCnt * b + ancInfo.nOffset +
						ancPos.y * ancInfo.cells.width + ancPos.x;
				pNegMask[iAncInBatch] = 0;
				auto nAncOff = std::find(posIndices.begin(), posIndices.end(),
						iAncInBatch) - posIndices.begin();
				if (nAncOff == posIndices.size()) { // if not exists, push new
					xyTruth.resize(xyTruth.size() + 2);
					xyScale.resize(xyScale.size() + 2);
					whTruth.resize(whTruth.size() + 2);
					whScale.resize(whScale.size() + 2);
					probsTruth.resize(probsTruth.size() + nNumClasses, 0);
					posIndices.push_back(iAncInBatch);
				}
				__CalcAnchorCoordTruthScale(truBox, batchImgSize[b], ancInfo,
						&xyTruth[nAncOff * 2], &xyScale[nAncOff * 2],
						&whTruth[nAncOff * 2], &whScale[nAncOff * 2]
						);
				probsTruth[nAncOff * nNumClasses + truth.second] = 1;
			}
		}
		int64_t nNumPosAncs = (int64_t)posIndices.size();
		auto optNoGrad = torch::TensorOptions(torch::requires_grad(false));
		auto tPosIndices = torch::from_blob(posIndices.data(),
				{nNumPosAncs}, optNoGrad.dtype(torch::kInt64)).clone();

		auto tSelAncs = torch::index_select(tPredVals, 0, tPosIndices);
		auto tPosXY = tSelAncs.slice(1, 0, 2);
		auto tPosWH = tSelAncs.slice(1, 2, 4);
		auto tPosConf = tSelAncs.slice(1, 8, 9).view(-1);
		auto tPosProbs = tSelAncs.slice(1, 9, nNumVals);
		auto tAllConf = tPredVals.slice(1, 8, 9).view(-1);

		auto tXYTruth = torch::from_blob(xyTruth.data(),
				{nNumPosAncs, 2}, optNoGrad).clone();
		auto tWHTruth = torch::from_blob(whTruth.data(),
				{nNumPosAncs, 2}, optNoGrad).clone();
		auto tXYScale = torch::from_blob(xyScale.data(),
				{nNumPosAncs, 2}, optNoGrad).clone();
		auto tWHScale = torch::from_blob(whScale.data(),
				{nNumPosAncs, 2}, optNoGrad).clone();
		auto tProbsTruth = torch::from_blob(probsTruth.data(),
				{nNumPosAncs, nNumClasses}, optNoGrad).clone();

		auto tXYLoss = tXYScale * m_BCELoss(tPosXY, tXYTruth);
		auto tWHLoss = tWHScale * m_MSELoss(tPosWH, tWHTruth);
		auto tNegConfLoss = m_BCELoss(tAllConf, torch::zeros_like(tAllConf,
				optNoGrad)) * tNegMask * m_fConfWeight;
		auto tPosConfLoss = m_BCELoss(tPosConf, torch::ones_like(tPosConf,
				optNoGrad));
		auto tProbsLoss = m_fProbWeight * m_BCELoss(tPosProbs, tProbsTruth);
		auto tLoss = tNegConfLoss.sum() + tPosConfLoss.sum()
				+ tProbsLoss.sum() + tXYLoss.sum() + tWHLoss.sum();

#ifdef DEBUG_DUMP_YOLO_DELTA
		auto tMaxNegConf = (tNegMask * tAllConf).amax();
		auto tMinPosConf = tPosConf.amin();
		auto tNegConfLossSum = tNegConfLoss.sum();
		auto tPosConfLossSum = tPosConfLoss.sum();
		auto nPosProbCnt = std::count(probsTruth.begin(), probsTruth.end(), 1);
		auto nNegProbCnt = std::count(probsTruth.begin(), probsTruth.end(), 0);
		auto tMinPosProb = (tPosProbs * tProbsTruth + (1 - tProbsTruth)).amin();
		auto tMaxNegProb = (tPosProbs * (1 - tProbsTruth)).amax();
		auto tXYLossSum = tXYLoss.sum(0);
		auto tWHLossSum = tWHLoss.sum(0);
		auto tProbsLossSum = tProbsLoss.sum();
		std::ostringstream oss;
		oss << std::setprecision(5) << std::fixed << std::left << tLoss.item<float>();
		oss << "\nNegConf: " << std::setw(8) << tMaxNegConf.item<float>()
			<< std::setw(8) << tNegConfLossSum.item<float>();
		oss << "\nPosConf: " << std::setw(8) << tMinPosConf.item<float>()
			<< std::setw(8) << tPosConfLossSum.item<float>();
		oss << "\nProbs: " << tMinPosProb.item<float>() << " vs "
				<< tMaxNegProb.item<float>() << ": " << tProbsLossSum.item<float>();
		// oss << "\nX: " << tPosXY[0][0].item<float>() << "->" << xyTruth[0] << " * "
		// 		<< tXYScale[0][0].item<float>() << ": " << tXYLossSum[0].item<float>();
		// oss << "\nY: " << tPosXY[0][1].item<float>() << "->" << xyTruth[1] << " * "
		// 		<< tXYScale[0][1].item<float>() << ": " << tXYLossSum[1].item<float>();
		// oss << "\nW: " << tPosWH[0][0].item<float>() << "->" << whTruth[0] << " * "
		// 		<< tWHScale[0][0].item<float>() << ": " << tWHLossSum[0].item<float>();
		// oss << "\nH: " << tPosWH[0][1].item<float>() << "->" << whTruth[1] << " * "
		// 		<< tWHScale[0][1].item<float>() << ": " << tWHLossSum[1].item<float>();
		LOG(INFO) << oss.str();
#endif
		return tLoss;
	}

	std::pair<ANCHOR_INFO, cv::Point> __GetBestAnchor(
			const cv::Rect2f &truBox, const cv::Size &imgSize) {
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
		cv::Point ancPos = {int(truCenter.x * bestAnc.cells.width),
							int(truCenter.y * bestAnc.cells.height)};
		return std::make_pair(bestAnc, ancPos);
	}

	void __CalcAnchorCoordTruthScale(const cv::Rect2f &truBox, cv::Size imgSize,
			const ANCHOR_INFO &ancInfo, float *pXYTruth, float *pXYScale,
			float *pWHTruth, float *pWHScale) {
		auto truCenter = (truBox.tl() + truBox.br()) / 2;
		pXYTruth[0] = truCenter.x * ancInfo.cells.width;
		pXYTruth[1] = truCenter.y * ancInfo.cells.height;
		pXYTruth[0] -= (int64_t)pXYTruth[0];
		pXYTruth[1] -= (int64_t)pXYTruth[1];
		pXYScale[0] = pXYScale[1] = m_fXYWeight * (2 - truBox.area());

		auto ancSize = ancInfo.boxSize;
		ancSize.width /= imgSize.width;
		ancSize.height /= imgSize.height;
		pWHTruth[0] = std::log(truBox.width / ancSize.width);
		pWHTruth[1] = std::log(truBox.height / ancSize.height);
		pWHScale[0] = pWHScale[1] = m_fSizeWeight * (2 - truBox.area());
	}
};

REGISTER_CREATOR(BasicLoss, YOLOLoss, "YOLO");
