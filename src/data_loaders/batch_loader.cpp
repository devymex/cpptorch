#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>
#include <glog/logging.h>

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

bool BatchLoader::GetBatch(size_t nBatchSize, torch::Device device,
			torch::Tensor &tData, torch::Tensor &tTarget) {
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
	if (m_tLoadingData.dim() == 0 || nBatchSize != m_tLoadingData.size(0)) {
		_LoadBatch(_GetBatchIndices(nBatchSize), device,
				m_tLoadingData, m_tLoadingTarget);
	}
	m_nCursor += nBatchSize;
	std::swap(m_tLoadingData, tData);
	std::swap(m_tLoadingTarget, tTarget);

	m_Worker = std::thread(
		[&](std::vector<size_t> _indices, torch::Device device,
				torch::Tensor &_tData, torch::Tensor &_tTarget) {
			_LoadBatch(std::move(_indices), device, _tData, _tTarget);
		}, _GetBatchIndices(nBatchSize), device,
			std::ref(m_tLoadingData), std::ref(m_tLoadingTarget));
	return true;
}

std::vector<size_t> BatchLoader::_GetBatchIndices(size_t nBatchSize) const {
	std::vector<size_t> indices;
	for (size_t i = 0; i < nBatchSize; ++i) {
		indices.push_back(m_Indices[(i + m_nCursor) % Size()]);
	}
	return indices;
}