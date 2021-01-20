#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "../argman.hpp"
#include "../utils.hpp"
#include "batch_loader.hpp"

BatchLoader::~BatchLoader() {
	if (m_Worker.joinable()) {
		m_Worker.join();
	}
}

void BatchLoader::Initialize(const nlohmann::json &jConf) {
	ArgMan argMan;
	Arg<bool> argShuffle("shuffle", false, argMan);
	ParseArgsFromJson(jConf, argMan);
	m_bShuffle = argShuffle();
}

void BatchLoader::ResetCursor() {
	if (m_Worker.joinable()) {
		m_Worker.join();
	}
	CHECK_NE(Size(), 0);
	if (m_Indices.size() != Size()) {
		m_Indices.resize(Size());
		std::iota(m_Indices.begin(), m_Indices.end(), 0);
	}
	if (m_bShuffle) {
		std::shuffle(m_Indices.begin(), m_Indices.end(), GetRG());
		m_nCursor = 0;
	} else {
		for (; m_nCursor >= Size(); ) {
			m_nCursor %= Size();
		}
	}
}

bool BatchLoader::GetBatch(uint64_t nBatchSize, TENSOR_ARY &data,
		TENSOR_ARY &targets, torch::Device device) {
	if (m_Worker.joinable()) {
		m_Worker.join();
	}
	CHECK_NE(Size(), 0);
	if (m_Indices.size() != Size()) {
		ResetCursor();
	}
	if (m_nCursor >= Size()) {
		return false;
	}
	if (m_LoadingData.empty() || nBatchSize != m_LoadingData[0].size(0)) {
		__LoadBatchToDevice(_GetBatchIndices(nBatchSize),
				m_LoadingData, m_LoadingTarget, device);
	}
	m_nCursor += nBatchSize;
	data.swap(m_LoadingData);
	targets.swap(m_LoadingTarget);

#ifdef DEBUG_DRAW_IMAGE_AND_LABEL
	auto tBoxes = targets[0].to(torch::kCPU);
	auto tImgSize = targets[1].to(torch::kCPU);
	auto tData = data[0].squeeze(0).permute({1, 2, 0}) * 255;
	tData = tData.to(torch::kCPU, torch::kUInt8);
	cv::Size imgSize(tImgSize[0].item<float>(), tImgSize[1].item<float>());
	cv::Mat img(imgSize, CV_8UC3, tData.data_ptr<uint8_t>());
	float *pBoxes = tBoxes.data_ptr<float>();
	for (uint64_t i = 0; i < tBoxes.size(0); ++i) {
		cv::Rect box = {cv::Point(int(pBoxes[i * 5 + 1] * imgSize.width), int(pBoxes[i * 5 + 2] * imgSize.height)),
						cv::Point(int(pBoxes[i * 5 + 3] * imgSize.width), int(pBoxes[i * 5 + 4] * imgSize.height))};
		cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite("test/img.png", img);
#endif

	m_Worker = std::thread(
		[&](std::vector<uint64_t> _indices, TENSOR_ARY &_data,
				TENSOR_ARY &_targets, torch::Device device) {
			__LoadBatchToDevice(std::move(_indices), _data, _targets, device);
		}, _GetBatchIndices(nBatchSize), std::ref(m_LoadingData),
				std::ref(m_LoadingTarget), device);
	return true;
}

std::vector<uint64_t> BatchLoader::_GetBatchIndices(uint64_t nBatchSize) const {
	std::vector<uint64_t> indices;
	for (uint64_t i = 0; i < nBatchSize; ++i) {
		indices.push_back(m_Indices[(i + m_nCursor) % Size()]);
	}
	return indices;
}

void BatchLoader::__LoadBatchToDevice(std::vector<uint64_t> indices,
		TENSOR_ARY &data, TENSOR_ARY &targets, torch::Device device) {
	_LoadBatch(std::move(indices), data, targets);
	for (auto &d : data) {
		if (d.device() != device) {
			d = d.to(device);
		}
	}
	for (auto &t : targets) {
		if (t.device() != device) {
			t = t.to(device);
		}
	}
}
