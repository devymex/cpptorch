#ifndef __BATCH_LOADER_HPP
#define __BATCH_LOADER_HPP

#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <torch/torch.h>
#include "../json.hpp"

class BatchLoader {
public:
	BatchLoader() = default;
	~BatchLoader();

	virtual void Initialize(const nlohmann::json &jConf);
	virtual uint64_t Size() const = 0;
	virtual void ResetCursor();
	virtual bool GetBatch(uint64_t nBatchSize, torch::Device device,
			torch::Tensor &tData, torch::Tensor &tTarget);

protected:
	virtual std::vector<uint64_t> _GetBatchIndices(uint64_t nBatchSize) const;
	virtual void _LoadBatch(std::vector<uint64_t> indices,
			torch::Tensor &tData, torch::Tensor &tTarget) = 0;

private:
	void __LoadBatch(std::vector<uint64_t> indices, torch::Device device,
			torch::Tensor &tData, torch::Tensor &tTarget);

private:
	uint64_t m_nCursor = 0;
	bool m_bShuffle = false;
	std::vector<uint64_t> m_Indices;
	std::thread m_Worker;
	torch::Tensor m_tLoadingData = torch::empty({});
	torch::Tensor m_tLoadingTarget = torch::empty({});
};

#endif //__BATCH_LOADER_HPP
