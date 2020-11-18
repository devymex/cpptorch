#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <random>
#include <string>
#include <vector>
#include "json.hpp"

std::mt19937& GetRG(uint32_t nSeed = 0);

bool LoadFileContent(const std::string &strFn, std::string &strFileBuf);

std::vector<std::string> EnumerateFiles(const std::string &strPath,
		const std::string &strPattern = "", bool bRecursive = true);

nlohmann::json LoadJsonFile(const std::string &strFilename);

#endif // _UTILS_HPP