#include <ctime>
#include <fstream>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "weights.hpp"
#include "utils.hpp"
#include "argman.hpp"
#include "json.hpp"
#include "creator.hpp"
#include "data_loaders/batch_loader.hpp"
#include "models/basic_model.hpp"
#include "loss_functions/basic_loss.hpp"

namespace bfs = boost::filesystem;
namespace tfunc = torch::nn::functional;

std::pair<uint64_t, std::string> FindLastState(const std::string &strPath,
		 const std::string &strName) {
	std::string strPrefix = "^" + strName + "_";
	std::string strNumber = "e([1-9][0-9]*)";
	std::string strSuffix = "\\.pkl\\.2[0-9]{11}$";
	std::string strPattern = strPrefix + strNumber + strSuffix;
	std::pair<uint64_t, std::string> ret;
	auto filenames = EnumerateFiles(strPath, strPattern);
	if (!filenames.empty()) {
		std::vector<uint64_t> epochs;
		for (const auto &strFn : filenames) {
			std::string strLeaf = bfs::path(strFn).leaf().string();
			boost::smatch match;
			boost::regex_search(strLeaf, match, boost::regex(strPattern));
			CHECK_EQ(match.size(), 2);
			auto strEpoch = std::string(match[1].first, match[1].second);
			epochs.push_back(std::stoi(strEpoch));
		}
		auto iMax = std::max_element(epochs.begin(), epochs.end());
		ret.first = *iMax;
		ret.second = filenames[iMax - epochs.begin()];
		CHECK_GT(ret.first, 0);
	}
	return ret;
}

std::string MakeStateFilename(const std::string &strPath,
		const std::string &strModelName, uint64_t nEpoch) {
	CHECK_GT(nEpoch, 0);
	std::time_t now = std::time(nullptr);
	std::ostringstream ossFilename;
	ossFilename << strModelName << "_e" << nEpoch << ".pkl."
			<< std::put_time(std::localtime(&now), "%y%m%d%OH%OM%OS");
	bfs::path savePath(strPath);
	return (savePath / ossFilename.str()).string();
}

int main(int nArgCnt, const char *ppArgs[]) {
	FLAGS_alsologtostderr = 1;
	google::InitGoogleLogging(ppArgs[0]);

	// Arguments Definitions
	// -------------------------------------------------------------------------
	Arg<std::string> argConfName("name");
	Arg<std::string> argDevice("device");
	Arg<std::string> argMode("mode", "TRAIN");
	Arg<std::string> argTestResultsFile("test_result_file");
	Arg<uint64_t> argDeviceID("device_id", 0);
	Arg<std::string> argDataRoot("data_root");
	Arg<std::string> argTrainList("train_list");
	Arg<std::string> argTestList("test_list");
	Arg<std::string> argModelFile("model_file");
	Arg<bool> argLabelBalance("label_balance", true);
	Arg<int32_t> argRandomSeed("random_seed", -1);
	Arg<uint64_t> argMaxEpoch("max_epoch", 10);
	Arg<uint64_t> argBatchSize("batch_size", 32);
	Arg<float> argLearningRate("learning_rate", 0.01f);
	Arg<uint64_t> argLRStepSize("lr_step_epochs", 0);
	Arg<float> argLRStepGamma("lr_step_gamma", 0.f);
	Arg<float> argWeightDecay("weight_decay", 0.f);
	Arg<float> argMomentum("momentum", 0.f);
	Arg<std::string> argLogPath("log_path");
	Arg<uint64_t> argLogIters("log_iters", 0);
	Arg<std::string> argStatePath("state_path");
	Arg<uint64_t> argStateSaveEpochs("save_state_epochs", 0);
	Arg<bool> argLoadLast("load_last", false);
	Arg<nlohmann::json> argModel("model");
	Arg<nlohmann::json> argLoss("loss");
	Arg<nlohmann::json> argTrainData("train_data");
	Arg<nlohmann::json> argTestData("test_data");

	// Configure Parsing and Arguments Checking
	// -------------------------------------------------------------------------
	CHECK_GT(nArgCnt, 1);
	auto jConf = LoadJsonFile(ppArgs[1]);
	ParseArgsFromJson(jConf);
	if (argConfName().empty()) {
		bfs::path confFile(ppArgs[1]);
		argConfName.Set(confFile.leaf().stem().string());
	}
	if (nArgCnt > 2) {
		auto jConfExt = nlohmann::json::parse(ppArgs[2]);
		ParseArgsFromJson(jConfExt);
		for (auto jItem : jConfExt.items()) {
			jConf[jItem.key()] = jItem.value();
		}
	}
	if (argLoadLast() || argStateSaveEpochs() != 0) {
		std::string strMsg = "load_last_epoch is true or save_epoch != 0, ";
		CHECK(!argStatePath().empty()) << strMsg << "but state_path is empty";
		boost::filesystem::path savePath(argStatePath());
		CHECK(boost::filesystem::exists(savePath)) << strMsg
				 << "but state_path \"" << argStatePath() << "\" not exists";
	}
	bool bTrainMode = argMode() == "TRAIN";
	if (!bTrainMode) {
		CHECK(argMode() == "TEST" || argMode() == "TRAIN");
	}

	// Log Subsystem Initialization
	// -------------------------------------------------------------------------
	if (!argLogPath().empty()) {
		bfs::path logPath(argLogPath());
		CHECK(bfs::is_directory(logPath));
		std::string strLeafName = argConfName() + ".log.";
		std::string strLogBaseName = (logPath / strLeafName).string();
		for (int i = 0; i < 4; ++i) {
			google::SetLogDestination(i, strLogBaseName.c_str());
			google::SetLogSymlink(i, "");
		}
	}
	LOG(INFO) << "Config File:\n" << jConf.dump(4);

	// Random Seed
	// -------------------------------------------------------------------------
	if (argRandomSeed() < 0) {
		std::random_device rd;
		argRandomSeed.Set(rd());
	}
	GetRG(argRandomSeed());
	torch::manual_seed(GetRG()());

	// Device Initialization
	// -------------------------------------------------------------------------
	torch::Device device = torch::kCPU;
	if (argDevice() != "CPU") {
		if (argDevice() == "GPU" || argDevice() == "CUDA") {
			device = torch::Device(torch::kCUDA, argDeviceID());
		} else {
			LOG(FATAL) << "Unrecognized device: " << argDevice();
		}
	}

	// Data Loader Preparation
	// -------------------------------------------------------------------------
	auto pTrainLdr = Creator<BatchLoader>::Create(argTrainData());
	auto pTestLdr = Creator<BatchLoader>::Create(argTestData());

	// Module Preparation
	// -------------------------------------------------------------------------
	uint64_t nInitEpoch = 0;
	auto pModel = Creator<BasicModel>::Create(argModel());
	pModel->SetDevice(device);
	if (argLoadLast()) {
		auto lastState = FindLastState(argStatePath(), argConfName());
		nInitEpoch = lastState.first;
		pModel->LoadWeights(argModelFile());
		if (nInitEpoch > 0) {
			LOG(INFO) << "Weights loaded from last epoch " << nInitEpoch;
		}
	}
	if (!bTrainMode) {
		argMaxEpoch.Set(nInitEpoch + 1);
		CHECK(!argTestResultsFile().empty());
	}
	auto pLoss = Creator<BasicLoss>::Create(argLoss());

	// Optimizer Initialization
	// -------------------------------------------------------------------------
	std::vector<torch::Tensor> parameters;
	for (const auto &pair : pModel->NamedParameters()) {
		parameters.push_back(pair.second);
	}
	float fLearningRate = argLearningRate();
	if (argLRStepSize() > 0) {
		uint64_t nSteps = nInitEpoch / argLRStepSize();
		fLearningRate *= std::pow(argLRStepGamma(), (float)nSteps);
	}
	using SGDOPTION = torch::optim::SGDOptions;
	torch::optim::SGD sgdOptim(parameters, SGDOPTION(fLearningRate)
			.weight_decay(argWeightDecay())
			.momentum(argMomentum()));
	// torch::Tensor tData, tTarget;
	// for (uint64_t nIter = 1; pTrainLdr->GetBatch(
	// 		argBatchSize(), device, tData, tTarget); ++nIter) {
	// 	LOG(INFO) << nIter;
	// }

	// Main Loop
	// -------------------------------------------------------------------------
	for (uint64_t nEpoch = 1; nEpoch + nInitEpoch <= argMaxEpoch(); ++nEpoch) {
		torch::Tensor tData, tTarget;
		auto &sgdOption = static_cast<SGDOPTION&>(sgdOptim.defaults());
		// Train phase
		if (argLRStepSize() > 0 && (nEpoch + nInitEpoch) % argLRStepSize() == 0) {
			uint64_t nSteps = (nEpoch + nInitEpoch) / argLRStepSize();
			float fDecay = std::pow(argLRStepGamma(), (float)nSteps);
			sgdOption.lr(argLearningRate() * fDecay);
		}
		pTrainLdr->ResetCursor();
		pModel->TrainMode(true);
		float fTrainLossSum = 0.f;
		for (uint64_t nIter = 1; bTrainMode && pTrainLdr->GetBatch(
				argBatchSize(), device, tData, tTarget); ++nIter) {
			sgdOptim.zero_grad();
			torch::Tensor tOutput = pModel->Forward({tData});
			float fLoss = pLoss->Backward(tOutput, tTarget);
			fTrainLossSum += fLoss;
			sgdOptim.step();
			if (argLogIters() > 0 && nIter % argLogIters() == 0) {
				LOG(INFO) << "train_iter=" << nIter << ", loss=" << fLoss;
			}
		}
		// Test phase
		pTestLdr->ResetCursor();
		pModel->TrainMode(false);
		float fTestLossSum = 0.f;
		std::vector<std::pair<int64_t, std::vector<float>>> testResults;
		for (uint64_t nIter = 1; pTestLdr->GetBatch(
				argBatchSize(), device, tData, tTarget); ++nIter) {
			torch::Tensor tOutput = pModel->Forward({tData});
			uint64_t nEndIdx = nIter * argBatchSize();
			if (nEndIdx > pTestLdr->Size()) {
				int64_t nValidCnt = pTestLdr->Size()
						- (nIter - 1) * argBatchSize();
				tOutput = tOutput.slice(0, 0, nValidCnt);
				tTarget = tTarget.slice(0, 0, nValidCnt);
			}
			float fLoss = pLoss->Evaluate(tOutput, tTarget);
			fTestLossSum += fLoss;
			if (argLogIters() > 0 && nIter % argLogIters() == 0) {
				LOG(INFO) << "test_iter=" << nIter << ", loss=" << fLoss;
			}
		}

		// Logging phase
		std::string strTrainLoss = !bTrainMode ? "N/A"
				: std::to_string(fTrainLossSum / pTrainLdr->Size());
		std::string strEpoch = !bTrainMode ? "TEST"
				: std::to_string(nEpoch + nInitEpoch);
		LOG(INFO) << "epoch " << strEpoch << ": lr=" << sgdOption.lr()
				  << ", train=" << strTrainLoss
				  << ", test=" << fTestLossSum / pTestLdr->Size()
		 		  << ", result: " << pLoss->FlushResults();

		if (argStateSaveEpochs() != 0 && (nEpoch + nInitEpoch)
				% argStateSaveEpochs() == 0) {
			pModel->SaveWeights(MakeStateFilename(argStatePath(),
					argConfName(), nEpoch + nInitEpoch));
		}
	}
}
