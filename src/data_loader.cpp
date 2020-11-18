#include <glog/logging.h>
#include "data_loader.hpp"

std::unique_ptr<BatchLoader> CreateDataLoader(const nlohmann::json &jConf) {
	std::unique_ptr<BatchLoader> pLoader;
	auto iType = jConf.find("loader");
	CHECK(iType != jConf.end());
	CHECK(iType->is_string());
	std::string strType = iType->get<std::string>();
	CHECK(!strType.empty());
	if (strType == "MNIST") {
		pLoader.reset(new MNISTLoader(jConf));
	} else {
		LOG(FATAL) << "Invalid type: " << strType;
	}
	return pLoader;
}
