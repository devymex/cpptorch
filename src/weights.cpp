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

template<typename _IS, typename _Elem>
uint64_t LoadBuffer(_IS &inStream, std::vector<_Elem> &buf) {
	uint64_t nBufLen = 0;
	CHECK(inStream.read((char*)&nBufLen, sizeof(nBufLen)));
	CHECK_EQ(nBufLen % sizeof(_Elem), 0);
	buf.resize(nBufLen / sizeof(_Elem));
	CHECK(inStream.read((char*)buf.data(), nBufLen));
	return nBufLen;
}

template<typename _OS, typename _Elem>
uint64_t SaveBuffer(_OS &outStream, const std::vector<_Elem> &buf) {
	uint64_t nBufLen = buf.size() * sizeof(_Elem);
	CHECK(outStream.write((char*)&nBufLen, sizeof(nBufLen)));
	if (nBufLen > 0) {
		CHECK(outStream.write((char*)buf.data(), nBufLen));
	}
	return nBufLen;
}

std::pair<std::string, std::vector<char>> LoadNamedBuffer(std::ifstream &inStream) {
	std::vector<char> name, buf;
	CHECK_GT(LoadBuffer(inStream, name), 0);
	CHECK_GT(LoadBuffer(inStream, buf), 0);
	std::string strName(name.begin(), name.end());
	return std::make_pair(std::move(strName), std::move(buf));
}

void SaveNamedBuffer(std::ofstream &outStream, const std::string &strName,
		const std::vector<char> &buf) {
	std::vector<char> name(strName.begin(), strName.end());
	CHECK_GT(SaveBuffer(outStream, name), 0);
	CHECK_GT(SaveBuffer(outStream, buf), 0);
}

NAMED_PARAMS LoadWeights(const std::string &strFilename) {
	std::ifstream inFile(strFilename);
	CHECK(inFile.is_open());
	NAMED_PARAMS namedParams;
	for (; inFile.peek() != EOF ;) {
		auto [strName, buf] = LoadNamedBuffer(inFile);
		auto tensor = torch::pickle_load(std::move(buf)).toTensor();
		namedParams[std::move(strName)] = std::move(tensor);
	}
	return namedParams;
}

void SaveWeights(const NAMED_PARAMS &weights, const std::string &strFilename) {
	std::ofstream outFile(strFilename);
	CHECK(outFile.is_open());
	for (const auto &param : weights) {
		auto buf = torch::pickle_save(param.second);
		SaveNamedBuffer(outFile, param.first, buf);
	}
}

bool InitModuleWeight(const std::string &strModuleType,
		NAMED_PARAMS &weights, NAMED_PARAMS &buffers) {
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
		auto iMean = buffers.find("running_mean");
		if (iMean != buffers.end()) {
			iMean->second.fill_(0);
		}
		auto iVar = buffers.find("running_var");
		if (iVar != buffers.end()) {
			iVar->second.fill_(1);
		}
		torch::nn::init::constant_(iWeight->second, 1.);
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
	} else if (strModuleType.find("YoloLayer") != std::string::npos) {
	} else {
		return false;
	}
	return true;
}
