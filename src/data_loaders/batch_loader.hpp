#ifndef __BATCH_LOADER_HPP
#define __BATCH_LOADER_HPP

#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <torch/torch.h>
#include "../types.hpp"
#include "../json.hpp"

class BatchLoader {
public:
	BatchLoader() = default;
	~BatchLoader();

	virtual void Initialize(const nlohmann::json &jConf);
	virtual uint64_t Size() const = 0;
	virtual void ResetCursor();
	virtual bool GetBatch(uint64_t nBatchSize, TENSOR_ARY &data,
			TENSOR_ARY &targets, torch::Device device);

protected:
	virtual std::vector<uint64_t> _GetBatchIndices(uint64_t nBatchSize) const;
	virtual void _LoadBatch(std::vector<uint64_t> indices,
			TENSOR_ARY &data, TENSOR_ARY &targets) = 0;

private:
	void __LoadBatchToDevice(std::vector<uint64_t> indices,
			TENSOR_ARY &data, TENSOR_ARY &targets, torch::Device device);

private:
	uint64_t m_nCursor = 0;
	bool m_bShuffle = false;
	std::vector<uint64_t> m_Indices;
	std::thread m_Worker;
	TENSOR_ARY m_LoadingData;
	TENSOR_ARY m_LoadingTarget;
};

#endif //__BATCH_LOADER_HPP
