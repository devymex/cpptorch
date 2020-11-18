#include <memory>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <glog/logging.h>
#include "utils.hpp"

std::mt19937& GetRG(uint32_t nSeed) {
	static std::mt19937 rg(nSeed);
	return rg;
}

std::vector<std::string> EnumerateFiles(const std::string &strPath,
		const std::string &strPattern, bool bRecursive) {
	namespace bfs = boost::filesystem;
	std::vector<std::string> filenames;
	for (auto iter = bfs::recursive_directory_iterator(strPath);
			iter != bfs::recursive_directory_iterator(); ++iter) {
		std::string strFilename = iter->path().filename().string();
		boost::smatch match;
		if (boost::regex_match(strFilename, match, boost::regex(strPattern))) {
			filenames.emplace_back(iter->path().string());
		}
	}
	return filenames;
}

bool LoadFileContent(const std::string &strFn, std::string &strFileBuf) {
	std::ifstream inFile(strFn, std::ios::binary);
	if (!inFile.is_open()) {
		return false;
	}
	inFile.seekg(0, std::ios::end);
	strFileBuf.resize((uint64_t)inFile.tellg());
	inFile.seekg(0, std::ios::beg);
	inFile.read(const_cast<char*>(strFileBuf.data()), strFileBuf.size());
	CHECK(inFile.good());
	return true;
}

nlohmann::json LoadJsonFile(const std::string &strFilename) {
	std::string strConfContent;
	CHECK(LoadFileContent(strFilename, strConfContent)) << strFilename;
	return nlohmann::json::parse(strConfContent);
}