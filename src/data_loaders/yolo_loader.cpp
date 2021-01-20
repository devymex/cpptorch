#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "../creator.hpp"
#include "../argman.hpp"
#include "../utils.hpp"
#include "batch_loader.hpp"

namespace bfs = boost::filesystem;
using STREAM_ITER = std::istream_iterator<float>;

class YOLOLoader : public BatchLoader {
public:
	void Initialize(const nlohmann::json &jConf) override {
		BatchLoader::Initialize(jConf);

		ArgMan argMan;
		Arg<std::string> argDataRoot("image_list", argMan);
		Arg<std::vector<int>> argOutputSize("output_size",
				std::vector<int>{416, 416}, argMan);
		Arg<uint64_t> argMaxTruths("max_truths", 200, argMan);
		ParseArgsFromJson(jConf, argMan);

		CHECK_EQ(argOutputSize().size(), 2);
		m_OutSize = cv::Size(argOutputSize()[0], argOutputSize()[1]);
		CHECK_GT(m_OutSize.area(), 0);
		CHECK_GT(m_OutSize.width, 0);

		CHECK_GT(argMaxTruths(), 0);
		m_nMaxTruths = argMaxTruths();

		CHECK(!argDataRoot().empty());
		std::ifstream listFile(argDataRoot());
		CHECK(listFile.is_open());
		for (std::string strLine; std::getline(listFile, strLine); ) {
			if (!strLine.empty()) {
				m_ImgList.emplace_back(std::move(strLine));
			}
		}
	}
	uint64_t Size() const override {
		return m_ImgList.size();
	}

protected:
	void _LoadBatch(std::vector<uint64_t> indices,
			TENSOR_ARY &data, TENSOR_ARY &targets) override {
		const uint64_t nTruthVals = 5; // c, x, y, w, h

		TENSOR_ARY images;
		TENSOR_ARY labels;
		for (uint64_t b = 0; b < indices.size(); ++b) {
			bfs::path imagePath(m_ImgList[indices[b]]);
			CHECK(imagePath.parent_path().leaf().string() == "JPEGImages");
			auto labelPath = imagePath.parent_path().parent_path() / "labels";
			labelPath /= (imagePath.stem().string() + ".txt");

			std::ifstream labelFile(labelPath.string());
			CHECK(labelFile.is_open());
			std::vector<float> labelBuf;
			std::copy(STREAM_ITER(labelFile), STREAM_ITER(),
					std::back_inserter(labelBuf));
			CHECK_EQ(labelBuf.size() % nTruthVals, 0);
			uint64_t nBoxCnt = labelBuf.size() / nTruthVals;
			for (uint64_t i = 0; i < nBoxCnt; ++i) {
				labelBuf[i * 5 + 1] -= labelBuf[i * 5 + 3] / 2; // cx -> x1
				labelBuf[i * 5 + 2] -= labelBuf[i * 5 + 4] / 2; // cy -> y1
				labelBuf[i * 5 + 3] += labelBuf[i * 5 + 1]; // w -> x2
				labelBuf[i * 5 + 4] += labelBuf[i * 5 + 2]; // y -> y2
			}
			if (nBoxCnt > m_nMaxTruths) {
				LOG(WARNING) << "Too many boxes in \"" << imagePath << "\"";
			}
			labelBuf.resize(m_nMaxTruths * nTruthVals, -1.f);
			auto tLabel = torch::from_blob(labelBuf.data(),
					{1, (long)m_nMaxTruths, (long)nTruthVals});
			labels.emplace_back(tLabel.clone());

			cv::Mat img = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
			cv::resize(img, img, m_OutSize);
			cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
			img.convertTo(img, CV_32F, 1.f / 255);

			auto tImage = torch::from_blob(img.data,
					{1, m_OutSize.height, m_OutSize.width, 3});
			images.emplace_back(tImage.permute({0, 3, 1, 2}).clone());
		}
		torch::Tensor tMeta = torch::zeros({2}, torch::TensorOptions(torch::kFloat32));
		tMeta.data_ptr<float>()[0] = (float)m_OutSize.width;
		tMeta.data_ptr<float>()[1] = (float)m_OutSize.height;
		data = TENSOR_ARY{torch::cat(images)};
		targets = TENSOR_ARY{torch::cat(labels), std::move(tMeta)};
	}

protected:
	uint64_t m_nMaxTruths;
	cv::Size m_OutSize;
	std::vector<std::string> m_ImgList;
};

REGISTER_CREATOR(BatchLoader, YOLOLoader, "YOLO");
