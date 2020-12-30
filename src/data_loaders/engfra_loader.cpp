#include <algorithm>
#include <fstream>
#include <memory>
#include <glog/logging.h>
#include "../json.hpp"
#include "../creator.hpp"
#include "../argman.hpp"
#include "../utils.hpp"
#include "batch_loader.hpp"

class EngFraLoader : public BatchLoader {
public:
	void Initialize(const nlohmann::json &jConf) override {
		BatchLoader::Initialize(jConf);

		ArgMan argMan;
		Arg<std::string> argDataFile("data_file", argMan);
		ParseArgsFromJson(jConf, argMan);

		std::string strFileContent;
		CHECK(LoadFileContent(argDataFile(), strFileContent)) << argDataFile();
	}

	uint64_t Size() const override {
			return m_EngFra.size();
	}
protected:
	void _LoadBatch(std::vector<uint64_t> indices,
			torch::Tensor &tData, torch::Tensor &tTarget) override {
	}

private:
	using SENTENCE = std::vector<uint64_t>;
	using PAIR_ENGFRA = std::pair<SENTENCE, SENTENCE>;
	std::vector<PAIR_ENGFRA> m_EngFra;
};

REGISTER_CREATOR(BatchLoader, EngFraLoader, "EngFraLang");
