#include <algorithm>
#include <array>
#include <sstream>
#include <vector>
#include <utility>
#include <boost/filesystem.hpp>

#include "../creator.hpp"
#include "../argman.hpp"
#include "../utils.hpp"
#include "batch_loader.hpp"

namespace bfs = boost::filesystem;
using namespace std::placeholders;
class LanguageLoader : public BatchLoader {
public:
	void Initialize(const nlohmann::json &jConf) override {
		BatchLoader::Initialize(jConf);

		ArgMan argMan;
		Arg<std::string> argDataRoot("data_root", argMan);
		Arg<uint64_t> argStrLen("string_length", 0, argMan);
		ParseArgsFromJson(jConf, argMan);

		m_nStrLen = argStrLen();
		bfs::path dataPath = bfs::path(argDataRoot()) / bfs::path("names");
		CHECK(bfs::is_directory(dataPath));
		std::fill(m_Charset.begin(), m_Charset.end(), 0);
		uint64_t nMaxLen = 0;
		for (auto &strFilename : EnumerateFiles(dataPath.string(), ".*\\.txt")) {
			std::string strFileContent;
			CHECK(LoadFileContent(strFilename, strFileContent));
			std::istringstream iss(strFileContent);
			for (std::string strLine; std::getline(iss, strLine); ) {
				m_Strings.emplace_back(strLine, (int64_t)m_Languages.size());
				std::for_each(strLine.begin(), strLine.end(), 
					[&](char c){m_Charset[Char2Index(c)] = 1;});
				nMaxLen = std::max(strLine.size(), nMaxLen);
			}
			m_Languages.emplace_back(bfs::path(strFilename).stem().string());
		}
		LOG(INFO) << "nMaxLen=" << nMaxLen;
		uint64_t nCharCnt = 0;
		for (auto &c : m_Charset) {
			if (c == 1) {
				c = ++nCharCnt;
			}
		}
	}

	uint64_t Size() const override {
		return m_Strings.size();
	}

protected:
	void _LoadBatch(std::vector<uint64_t> indices,
			torch::Tensor &tData, torch::Tensor &tTarget) override {
		uint64_t nBachSize = indices.size();
		CHECK_GT(nBachSize, 0);

		std::vector<std::string> strings;
		std::vector<int64_t> labels;
		uint64_t nStrLen = 0;
		for (auto i: indices) {
			strings.push_back(m_Strings[i].first);
			labels.push_back(m_Strings[i].second);
			nStrLen = m_nStrLen > 0 ? m_nStrLen:
					std::max(strings.back().size(), m_nStrLen);
		}
		CHECK_GT(nStrLen, 0);
		uint64_t nCharNum = m_Charset.size();
		std::vector<float> dataBuf(nBachSize * nStrLen * nCharNum, 0.f);
		for (uint64_t i = 0; i < strings.size(); ++i) {
			for (uint64_t j = 0; j < strings[i].size(); ++j) {
				uint64_t nOffset = i * nStrLen * nCharNum + j * nCharNum;
				dataBuf[nOffset + Char2Index(strings[i][j])] = 1.f;
			}
		}
		tData = torch::tensor(dataBuf).reshape({(int64_t)nBachSize,
				(int64_t)nStrLen, (int64_t)nCharNum});
		tTarget = torch::tensor(labels).reshape({(int64_t)nBachSize});
	}
private:
	inline uint64_t Char2Index(char c) {
		return *(uint8_t*)&c;
	}
	std::vector<std::pair<std::string, int64_t>> m_Strings;
	std::vector<std::string> m_Languages;
	std::array<uint64_t, 256> m_Charset;
	uint64_t m_nStrLen = 0;
	bool m_bVarLen = false;
};

REGISTER_CREATOR(BatchLoader, LanguageLoader, "Language");
