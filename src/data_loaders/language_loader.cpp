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
		ParseArgsFromJson(jConf, argMan);

		bfs::path dataPath = bfs::path(argDataRoot()) / bfs::path("names");
		CHECK(bfs::is_directory(dataPath));
		
		std::fill(m_Charset.begin(), m_Charset.end(), 0);
		for (auto &strFilename : EnumerateFiles(dataPath.string(), ".*\\.txt")) {
			std::string strFileContent;
			CHECK(LoadFileContent(strFilename, strFileContent));
			std::istringstream iss(strFileContent);
			for (std::string strLine; std::getline(iss, strLine); ) {
				m_Strings.emplace_back(strLine, (int64_t)m_Languages.size());
				std::for_each(strLine.begin(), strLine.end(), 
					[&](char c){m_Charset[(uint8_t)c] = 1;});
			}
			m_Languages.emplace_back(bfs::path(strFilename).stem().string());
		}
		uint32_t nCharCnt = 0;
		for (auto &c : m_Charset) {
			if (c == 1) {
				c = ++nCharCnt;
			}
		}
	}

	size_t Size() const override {
		return m_Strings.size();
	}

protected:
	void _LoadBatch(std::vector<size_t> indices,
			torch::Tensor &tData, torch::Tensor &tTarget) override {
		uint64_t nBachSize = indices.size();
		CHECK_GT(nBachSize, 0);

		std::vector<std::string> strings;
		std::vector<int64_t> labels;
		uint64_t nStrLen = 0;
		for (auto i: indices) {
			strings.push_back(m_Strings[i].first);
			labels.push_back(m_Strings[i].second);
			if (strings.back().size() > nStrLen) {
				nStrLen = strings.back().size();
			}
		}
		CHECK_GT(nStrLen, 0);

		uint64_t nCharNum = m_Charset.size();
		std::vector<float> dataBuf(nBachSize * nStrLen * nCharNum, 0.f);
		for (uint64_t i = 0; i < strings.size(); ++i) {
			uint64_t nOffset = i * nStrLen * nCharNum;
			for (uint64_t j = 0; j < strings[i].size(); ++j) {
				dataBuf[nOffset + j * nCharNum + strings[i][j]] = 1.f;
			}
		}
		tData = torch::tensor(dataBuf).reshape({(int32_t)nBachSize,
				(int32_t)nStrLen, 256});
		tTarget = torch::tensor(labels).reshape({(int32_t)nBachSize});
	}
private:
	std::vector<std::pair<std::string, int64_t>> m_Strings;
	std::vector<std::string> m_Languages;
	std::array<uint32_t, 256> m_Charset;
};

REGISTER_CREATOR(BatchLoader, LanguageLoader, "Language");
