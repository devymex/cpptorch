#ifndef __DATA_LOADER_HPP
#define __DATA_LOADER_HPP

#include <memory>

#include "data_loaders/mnist_loader.hpp"
#include "json.hpp"

std::unique_ptr<BatchLoader> CreateDataLoader(const nlohmann::json &jConf);

#endif // #ifndef __DATA_LOADER_HPP