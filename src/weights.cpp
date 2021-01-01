#include <ctime>
#include <string>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>

#include <torch/nn.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "utils.hpp"
#include "weights.hpp"

namespace bfs = boost::filesystem;

uint8_t HexChar2Num(uint8_t c) {
	if (isdigit((int)c)) {
		return c - '0';
	}
	if (c >= 'A' && c <= 'F') {
		return c - 'A' + 10;
	}
	return 0xFF;
}

uint8_t Num2HexChar(uint8_t n) {
	if (n < 10) {
		return n + '0';
	}
	if (n < 16) {
		return n + 'A' - 10;
	}
	return 0xFF;
}

uint8_t HexWord2Num(uint16_t c) {
	uint8_t lo = HexChar2Num((uint8_t)(c >> 8));
	uint8_t hi = HexChar2Num((uint8_t)(c & 0xFF));
	CHECK_NE(hi, 0xFF);
	CHECK_NE(lo, 0xFF);
	return (hi << 4) | lo;
}

uint16_t Num2HexWord(uint8_t n) {
	uint8_t lo = Num2HexChar((uint8_t)(n >> 4));
	uint8_t hi = Num2HexChar((uint8_t)(n & 0xF));
	return (uint16_t(hi << 8) | uint16_t(lo));
}

NAMED_PARAMS LoadWeights(const std::string &strFilename) {
	std::ifstream inFile(strFilename);
	CHECK(inFile.is_open());
	NAMED_PARAMS namedParams;
	for (std::string strLine; std::getline(inFile, strLine); ) {
		auto iNameEnd = strLine.find(' ');
		CHECK(iNameEnd != std::string::npos);
		CHECK_GT(strLine.size(), iNameEnd);
		const char *pData = strLine.data();
		std::vector<char> bytes;
		for (uint64_t i = iNameEnd + 2; i < strLine.size(); i += 2) {
			uint16_t hex = *(uint16_t*)(pData + i);
			bytes.push_back(char(HexWord2Num(hex)));
		}
		std::string strName(strLine.begin(), strLine.begin() + iNameEnd);
		namedParams[strName] = torch::pickle_load(bytes).toTensor();
	}
	return namedParams;
}

void SaveWeights(const NAMED_PARAMS &weights, const std::string &strFilename) {
	std::ofstream outFile(strFilename);
	CHECK(outFile.is_open());
	for (const auto &param : weights) {
		outFile << param.first << " ";
		auto tensor = param.second;
		if (!tensor.device().is_cpu()) {
			tensor = tensor.to(torch::kCPU);
		}
		auto tensorBytes = torch::pickle_save(tensor);
		for (auto b: tensorBytes) {
			uint16_t hex = Num2HexWord(b);
			outFile << ((char*)(&hex))[0] << ((char*)(&hex))[1];
		}
		outFile << std::endl;
	}
}

void Test() {
	std::string str;
	std::vector<char> ary = { -1, -2, -3, -4, -5};
	for (auto b: ary) {
		uint16_t hex = Num2HexWord(b);
		str.push_back(((char*)(&hex))[0]);
		str.push_back(((char*)(&hex))[1]);
	}
	ary.clear();
	auto pData = str.data();
	for (uint64_t i = 0; i < str.size(); i += 2) {
		uint16_t hex = *(uint16_t*)(pData + i);
		ary.push_back(char(HexWord2Num(hex)));
	}
}

bool InitModuleWeight(const std::string &strModuleType, NAMED_PARAMS &weights) {
	auto iWeight = weights.find("weight");
	auto iBias = weights.find("bias");
	if (strModuleType.find("Conv2d") != std::string::npos) {
		CHECK(iWeight != weights.end());
		torch::nn::init::xavier_normal_(iWeight->second);
		if (iBias != weights.end()) {
			torch::nn::init::normal_(iBias->second);
		}
	} else if (strModuleType.find("Linear") != std::string::npos) {
		CHECK(iWeight != weights.end());
		torch::nn::init::xavier_normal_(iWeight->second);
		if (iBias != weights.end()) {
			torch::nn::init::normal_(iBias->second);
		}
	} else if (strModuleType.find("BatchNorm2d") != std::string::npos) {
		CHECK(iWeight != weights.end());
		CHECK(iBias != weights.end());
		torch::nn::init::normal_(iWeight->second, 1., 0.02);
		torch::nn::init::constant_(iBias->second, 0.);
	} else if (strModuleType.find("LSTM") != std::string::npos) {
		for (auto &w : weights) {
			CHECK_GT(w.first.size(), 5);
			auto strPrefix = w.first.substr(0, 5);
			if (strPrefix == "bias_") {
				torch::nn::init::constant_(w.second, 0.);
			} else {
				CHECK(strPrefix == "weight");
				torch::nn::init::xavier_normal_(w.second);
			}
		}
	} else {
		return false;
	}
	return true;
}
