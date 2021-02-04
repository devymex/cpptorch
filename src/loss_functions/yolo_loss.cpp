#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>
#include "../creator.hpp"
#include "../argman.hpp"
#include "../utils.hpp"
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
	const std::vector<cv::Size2f> m_AncSize = {
		cv::Size2f(116,90), cv::Size2f(156,198), cv::Size2f(373,326),
		cv::Size2f(30,61),  cv::Size2f(62,45),   cv::Size2f(59,119),
		cv::Size2f(10,13),  cv::Size2f(16,30),   cv::Size2f(33,23)
	};
	float m_fXYWeight = 1.f;
	float m_fSizeWeight = 1.f;
	float m_fNegConfRatio = 1.f;
	float m_fProbWeight = 1.f;
	float m_fNegIgnoreIoU = 0.5f;
	float m_fNMSThres = 0.45f;
	float m_fConfThres = 0.24f;

	std::vector<ANCHOR_INFO> m_AncInfos;
	std::vector<std::string> m_ClsNames;
	uint64_t m_nImgAncCnt = 0; // amount of different size anchors in an image
	bool m_bPreview = false;

	torch::nn::MSELoss m_MSELoss;
	torch::nn::BCELoss m_BCELoss;
	torch::nn::BCELoss m_BCELossWeight;

public:
	void Initialize(const nlohmann::json &jConf) override {
		BasicLoss::Initialize(jConf);
		ArgMan argMan;
		Arg<float> argXYWeight("xy_weight", 1.0f, argMan);
		Arg<float> argNegConfRatio("neg_conf_ratio", 1.0f, argMan);
		Arg<float> argSizeWeight("size_weight", 1.0f, argMan);
		Arg<float> argProbWeight("prob_weight", 1.0f, argMan);
		Arg<float> argIgnoreIoU("ignore_iou", 0.5f, argMan);
		Arg<bool> argPreview("preview", false, argMan);
		Arg<std::string> argClassNames("class_names", argMan);
		ParseArgsFromJson(jConf, argMan);

		m_fXYWeight = argXYWeight();
		m_fNegConfRatio = argNegConfRatio();
		m_fSizeWeight = argSizeWeight();
		m_fProbWeight = argProbWeight();
		m_fNegIgnoreIoU = argIgnoreIoU();
		m_bPreview = argPreview();
		m_MSELoss->options.reduction() = torch::kNone; // for respective scales
		m_BCELoss->options.reduction() = torch::kNone;
		m_BCELossWeight->options.reduction() = torch::kNone;

		if (!argClassNames().empty()) {
			std::ifstream listFile(argClassNames());
			if (listFile.is_open()) {
				for (std::string strLine; std::getline(listFile, strLine); ) {
					if (!strLine.empty()) {
						m_ClsNames.emplace_back(std::move(strLine));
					}
				}
			}
		}
	}

	float Backward(TENSOR_ARY outputs, TENSOR_ARY targets) override {
		outputs[0].retain_grad();
		auto [tLoss, nNumAncs] = __CalcLoss(outputs, targets);
		tLoss = tLoss.sum();
		tLoss.backward();

		// for (int64_t i = 0; i < tConfGrad.size(0) / nBatchSize; ++i) {
		// 	for (int64_t j = 0; j < tConfGrad.size(1); ++j) {
		// 		float fGrad = pConfGrad[i * tConfGrad.size(1) + j];
		// 		float y = pAllVals[i * tConfGrad.size(1) + j];
		// 		if (j == 8 || (pConfMask[i] > 0 && (j > 8 || j < 2))) {
		// 			fGrad *= y * (1 - y);
		// 		} else if (j >=4 && j < 8) {
		// 			continue;
		// 		}
		// 		dumpFile << -fGrad << " ";
		// 	}
		// 	dumpFile << std::endl;
		// }
		// dumpFile.close();
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
		CHECK_GE(targets.size(), 2);
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
		if (m_bPreview) {
			__RenderResults(tPredVals.reshape({nBatchSize, -1, nNumVals}),
					targets[2]);
		}
		auto tLoss = __CalcLoss(tPredVals, truths, batchImgSize);
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
			const std::vector<std::vector<TRUTH>> &truths,
			const std::vector<cv::Size> &batchImgSize) {
		auto nBatchSize = truths.size();
		auto nNumBatchAncs = tPredVals.size(0);
		auto nNumVals = tPredVals.size(1);
		CHECK_EQ(nNumBatchAncs % nBatchSize, 0);

		auto tPredBoxes = tPredVals.slice(1, 4, 8).contiguous();
		auto tPredConf = tPredVals.slice(1, 8, 9).argmax(1).contiguous();
		auto tMaxProb = tPredVals.slice(1, 9, nNumVals).detach();
		auto pPredBoxes = tPredBoxes.data_ptr<float>(); // [x1, y1, x2, y2]
		auto pMaxProb = tMaxProb.data_ptr<float>();

		auto optNoGrad = torch::TensorOptions(torch::requires_grad(false));
		auto tNegMask = torch::ones({nNumBatchAncs},
				optNoGrad.dtype(torch::kInt32)).contiguous();
		auto pNegMask = tNegMask.data_ptr<int32_t>();
		for (uint64_t i = 0; i < nNumBatchAncs; ++i) {
			if (pMaxProb[i] > m_fNegProbThres) {
				auto pTLBR = (cv::Point2f*)(pPredBoxes + i * 4);
				cv::Rect2f predBox = { pTLBR[0], pTLBR[1] };
				auto fBestIoU = 0.f;
				for (auto &truth : truths[i / m_nImgAncCnt]) {
					fBestIoU = std::max(fBestIoU, IoU(predBox, truth.first));
				}
				if (fBestIoU > m_fNegIgnoreIoU) {
					pNegMask[i] = 0;
				}
			}
		}
		return tNegMask;
	}

	torch::Tensor __CalcLoss(torch::Tensor &tPredVals,
			const std::vector<std::vector<TRUTH>> &truths,
			const std::vector<cv::Size> &batchImgSize) {
		auto nBatchSize = truths.size();
		auto nNumBatchAncs = tPredVals.size(0);
		auto nNumVals = tPredVals.sizes().back();
		auto nNumClasses = nNumVals - 9;
		auto optNoGrad = torch::TensorOptions(torch::requires_grad(false));

		torch::Tensor tConfMask = __GetNegMask(tPredVals, truths, batchImgSize);
		torch::Tensor tConfTruth = torch::zeros_like(tConfMask,
				optNoGrad.dtype(torch::kFloat32)).contiguous();
		auto pConfMask = tConfMask.data_ptr<int32_t>();
		auto pConfTruth = tConfTruth.data_ptr<float>();
		std::vector<int64_t> posIndices;
		std::vector<float> xyTruth, xyScale, whTruth, whScale, probsTruth;
		for (uint64_t b = 0; b < nBatchSize; ++b) {
			for (auto &truth: truths[b]) {
				auto &truBox = truth.first;

				auto [ancInfo, ancPos] = __GetBestAnchor(truBox, batchImgSize[b]);
				auto iAncInBatch = m_nImgAncCnt * b + ancInfo.nOffset +
						ancPos.y * ancInfo.cells.width + ancPos.x;
				pConfMask[iAncInBatch] = 1;
				pConfTruth[iAncInBatch] = 1.f;

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
		auto tPosIndices = torch::from_blob(posIndices.data(),
				{nNumPosAncs}, optNoGrad.dtype(torch::kInt64)).clone();

		for (uint64_t b = 0; b < nBatchSize && m_fNegConfRatio > 0; ++b) {
			std::vector<int64_t> negIndices;
			for (int64_t i = 0; i < m_nImgAncCnt; ++i) {
				int64 idx = b * m_nImgAncCnt + i;
				if (pConfTruth[idx] == 0 && pConfMask[idx] == 1) {
					negIndices.push_back(i);
				}
			}
			int64_t nNegRemove = (int64_t)negIndices.size()
					- (int64_t)(truths[b].size() * m_fNegConfRatio);
			if (nNegRemove > 0) {
				std::shuffle(negIndices.begin(), negIndices.end(), GetRG());
				negIndices.resize(std::max(nNegRemove, int64_t(0)));
				for (auto &idx: negIndices) {
					pConfMask[b * m_nImgAncCnt + idx] = 0;
				}
			}
		}

		auto tSelAncs = torch::index_select(tPredVals, 0, tPosIndices);
		auto tPosXY = tSelAncs.slice(1, 0, 2);
		auto tPosWH = tSelAncs.slice(1, 2, 4);
		auto tPosProbs = tSelAncs.slice(1, 9, nNumVals);
		auto tAllConf = tPredVals.slice(1, 8, 9).reshape(-1);

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

		m_BCELossWeight->options.weight() = tConfMask;
		auto tXYLoss = tXYScale * m_BCELoss(tPosXY, tXYTruth);
		auto tWHLoss = tWHScale * m_MSELoss(tPosWH, tWHTruth);
		auto tProbsLoss = m_fProbWeight * m_BCELoss(tPosProbs, tProbsTruth);
		auto tConfLoss = m_BCELossWeight(tAllConf, tConfTruth);
		auto tLoss = tConfLoss.sum() + tProbsLoss.sum() + tXYLoss.sum() + tWHLoss.sum();

#ifdef DEBUG_DUMP_YOLO_DELTA
		float fMinPosConf = 1.0f, fMaxNegConf = 0.f;
		tAllConf = tAllConf.contiguous();
		auto pAllConf = tAllConf.data_ptr<float>();
		int64_t nPosCnt = 0, nNegCnt = 0;
		for (int64_t i = 0; i < nNumBatchAncs; ++i) {
			if (pConfMask[i] == 1) {
				if (pConfTruth[i]) {
					fMinPosConf = std::min(fMinPosConf, pAllConf[i]);
					++nPosCnt;
				} else {
					fMaxNegConf = std::max(fMaxNegConf, pAllConf[i]);
					++nNegCnt;
				}
			}
		}
		auto tXYLossSum = tXYLoss.sum(0);
		auto tWHLossSum = tWHLoss.sum(0);
		auto tConfLossSum = tConfLoss.sum();
		auto tProbsLossSum = tProbsLoss.sum();
		auto tMinPosProb = (tPosProbs * tProbsTruth + (1 - tProbsTruth)).amin();
		auto tMaxNegProb = (tPosProbs * (1 - tProbsTruth)).amax();
		auto nPosProbCnt = std::count(probsTruth.begin(), probsTruth.end(), 1);
		auto nNegProbCnt = std::count(probsTruth.begin(), probsTruth.end(), 0);
		std::ostringstream oss;
		oss << std::setprecision(5) << std::fixed << "loss: " << tLoss.item<float>();
		oss << "\nConf (" << tConfLossSum.item<float>() << "): " << fMinPosConf
			<< " vs " << fMaxNegConf << ", (" << nPosCnt << " vs " << nNegCnt << ")";
		oss << "\nProb (" << tProbsLossSum.item<float>() << "): " <<
			tMinPosProb.item<float>() << " vs " << tMaxNegProb.item<float>();
		oss << "\nX    (" << tXYLossSum[0].item<float>() << "): " << tPosXY[0][0].item<float>()
			<< " -> " << xyTruth[0] << ", " << tXYScale[0][0].item<float>();
		oss << "\nY    (" << tXYLossSum[1].item<float>() << "): " << tPosXY[0][1].item<float>()
			<< " -> " << xyTruth[1] << ", " << tXYScale[0][1].item<float>();
		oss << "\nW    (" << tWHLossSum[0].item<float>() << "): " << tPosWH[0][0].item<float>()
			<< " -> " << whTruth[0] << ", " << tWHScale[0][0].item<float>();
		oss << "\nH    (" << tWHLossSum[1].item<float>() << "): " << tPosWH[0][1].item<float>()
			<< " -> " << whTruth[1] << ", " << tWHScale[0][1].item<float>();
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

	struct BBOX {
		cv::Rect2f rect;
		float fConf;
		std::vector<float> probs;
	};
	
	void NMS(std::vector<BBOX> &boxes) {
		CHECK_GT(boxes.size(), 0);
		auto nNumCls = boxes[0].probs.size();
		for (uint64_t c = 0; c < nNumCls; ++c) {
			std::sort(boxes.begin(), boxes.end(),
				[&](const BBOX &b1, const BBOX &b2) {
					return b1.probs[c] > b2.probs[c];
				});
			for (uint64_t i = 0; i < boxes.size() - 1; ++i) {
				if (boxes[i].fConf > 0 && boxes[i].probs[c] > 0) {
					for (uint64_t j = i + 1; j < boxes.size(); ++j) {
						if (IoU(boxes[i].rect, boxes[j].rect) > m_fNMSThres) {
							boxes[j].probs[c] = 0;
						}
					}
				}
			}
		}
	}

	void __RenderResults(const torch::Tensor &tBatchPred,
			const torch::Tensor &tBatchImg) {
		cv::Size cellSize(416, 416);
		cv::Size cells(4, 2);
		cv::Mat cellPreview = cv::Mat::zeros(cellSize.height * cells.height,
				cellSize.width * cells.width, CV_8UC3);
		for (int64_t i = 0; i < tBatchPred.size(0) && i < cells.area(); ++i) {
			auto tImg = tBatchImg[i].detach().cpu().contiguous();
			auto imgShape = tImg.sizes();
			CHECK_EQ(imgShape[0], 3);
			auto pImg = tImg.data_ptr<float>();
			std::vector<cv::Mat> bgrChs;
			for (int64_t i = 0; i < tImg.size(0); ++i) {
				cv::Mat img(imgShape[1], imgShape[2], CV_32F,
						tImg[i].data_ptr<float>());
				img.convertTo(img, CV_8U, 255, 0);
				bgrChs.emplace_back(img.clone());
			}
			std::swap(bgrChs[0], bgrChs[2]);
			cv::Mat img;
			cv::merge(bgrChs, img);
			if (img.size() != cellSize) {
				cv::resize(img, img, cellSize);
			}
			
			std::vector<int64_t> shape = {-1, tBatchPred.sizes().back()};
			auto tPredValsCPU = tBatchPred[i].detach().cpu()
					.reshape(shape).contiguous();
			auto pPredVals = tPredValsCPU.data_ptr<float>();
			std::vector<BBOX> boxes;
			for (int64_t j = 0; j < tPredValsCPU.size(0); ++j) {
				auto pBoxVals = pPredVals + j * shape[1];
				BBOX box;
				if (pBoxVals[8] > m_fConfThres) {
					auto pPts = (cv::Point2f*)(pBoxVals + 4);
					box.rect = cv::Rect2f(pPts[0], pPts[1]);
					box.fConf = pBoxVals[8];
					for (int k = 9; k < shape[1]; ++k) {
						box.probs.push_back(pBoxVals[k] * box.fConf);
					}
					boxes.emplace_back(box);
				}
			}
			if (!boxes.empty() && boxes.size() < 500) {
				NMS(boxes);
				for (auto &b: boxes) {
					auto fMaxProb = *std::max_element(b.probs.begin(), b.probs.end());
					if (fMaxProb <= m_fConfThres) {
						b.fConf = 0;
					}
				}
				boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
					[](const BBOX &b){
						return b.fConf == 0;
					}), boxes.end());
				for (auto &b: boxes) {
					b.rect.x *= img.cols;
					b.rect.y *= img.rows;
					b.rect.width *= img.cols;
					b.rect.height *= img.rows;
					std::uniform_int_distribution<int> randClr(0, 255);
					cv::Scalar color(randClr(GetRG()), randClr(GetRG()),
							randClr(GetRG()));
					cv::rectangle(img, b.rect, color, 2);
					auto iCls = std::max_element(b.probs.begin(), b.probs.end())
							- b.probs.begin();
					std::string strClsName = std::to_string(iCls);
					if (m_ClsNames.size() > iCls) {
						strClsName = m_ClsNames[iCls];
					}
					cv::putText(img, strClsName, b.rect.tl(),
							cv::FONT_HERSHEY_PLAIN, 1.0, color);
				}
			}
			cv::Point beg((i % cells.width) * cellSize.width,
							(i / cells.width) * cellSize.height);
			img.copyTo(cellPreview(cv::Rect(beg, cellSize)));
		}
		cv::imshow("preview", cellPreview);
		cv::waitKey(10);
	}
};

REGISTER_CREATOR(BasicLoss, YOLOLoss, "YOLO");
