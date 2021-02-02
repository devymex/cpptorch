#include <ctime>
#include <fstream>
#include <iomanip>
#include <thread>

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
#include "loss_functions/basic_loss.hpp"
#include "optimizers/basic_optimizer.hpp"
#include "models/basic_model.hpp"

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
	Arg<uint64_t> argThreadNum("thread_num", 0);
	Arg<uint64_t> argAccGradIters("acc_grad_iters", 0);
	Arg<std::string> argLogPath("log_path");
	Arg<uint64_t> argLogIters("log_iters", 0);
	Arg<std::string> argStatePath("state_path");
	Arg<uint64_t> argStateSaveEpochs("save_state_epochs", 0);
	Arg<bool> argLoadLast("load_last", false);
	Arg<bool> argEpochLog("epoch_log", true);
	Arg<nlohmann::json> argTrainData("train_data");
	Arg<nlohmann::json> argTestData("test_data");
	Arg<nlohmann::json> argOptimizer("optimizer");
	Arg<nlohmann::json> argLoss("loss");
	Arg<nlohmann::json> argModel("model");

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
			CHECK(torch::cuda::is_available());
			device = torch::Device(torch::kCUDA, argDeviceID());
		} else {
			LOG(FATAL) << "Unrecognized device: " << argDevice();
		}
	}
	auto nNumThread = argThreadNum();
	if (nNumThread <= 0) {
		nNumThread = std::thread::hardware_concurrency();
	}
	torch::init_num_threads();
	torch::set_num_threads(nNumThread);

	// Data Loader Preparation
	// -------------------------------------------------------------------------
	auto pTrainLdr = Creator<BatchLoader>::Create(argTrainData());
	auto pTestLdr = Creator<BatchLoader>::Create(argTestData());

	// Module Preparation
	// -------------------------------------------------------------------------
	uint64_t nInitEpoch = 0;
	auto pModel = Creator<BasicModel>::Create(argModel());
	pModel->SetDevice(device);
	pModel->InitWeights(InitModuleWeight);
	if (argLoadLast()) {
		auto lastState = FindLastState(argStatePath(), argConfName());
		if (!lastState.second.empty()) {
			nInitEpoch = lastState.first;
			pModel->LoadWeights(lastState.second);
			if (nInitEpoch > 0) {
				LOG(INFO) << "Weights loaded from last epoch " << nInitEpoch;
			}
		}
	}
	if (!bTrainMode) {
		argMaxEpoch.Set(nInitEpoch + 1);
		CHECK(!argTestResultsFile().empty());
	}
	auto pLoss = Creator<BasicLoss>::Create(argLoss());

	// Optimizer Initialization
	// -------------------------------------------------------------------------
	auto pOptimizer = Creator<BasicOptimizer>::Create(argOptimizer());
	pOptimizer->SetModel(*pModel);

	// Main Loop
	// -------------------------------------------------------------------------
	for (uint64_t nEpoch = 1; nEpoch + nInitEpoch <= argMaxEpoch(); ++nEpoch) {
		TENSOR_ARY data, targets;
		pTrainLdr->ResetCursor();
		pModel->TrainMode(true);
		float fTrainLossSum = 0.f;
		for (uint64_t nIter = 1; bTrainMode && pTrainLdr->GetBatch(
				argBatchSize(), data, targets, device); ++nIter) {
			if (nIter == 1) {
				pOptimizer->ZeroGrad();
			}
			TENSOR_ARY outputs = pModel->Forward(std::move(data));
			float fLoss = pLoss->Backward(std::move(outputs), std::move(targets));
			fTrainLossSum += fLoss;
			// auto nb = pModel->NamedBuffers();
			// for (auto &np: pModel->NamedParameters()) {
			// 	if (np.first.find("conv.weight") != std::string::npos) {
			// 		auto x = np.second.grad().detach().reshape(-1).contiguous().cpu();
			// 		auto p = x.data_ptr<float>();
			// 		LOG(INFO) << np.first << " " << -p[0] << " " << -p[1] << " " << -p[2]
			// 				<< " " << -p[3] << " " << -p[4] << " " << -p[5];
			// 	}
			// }
			if (argAccGradIters() == 0 || (nIter % argAccGradIters() == 0)) {
				pOptimizer->IterStep();
				pOptimizer->ZeroGrad();
			}
			if (argLogIters() > 0 && nIter % argLogIters() == 0) {
				LOG(INFO) << "train_iter=" << nIter
						  << ", loss=" << fLoss / argBatchSize();
			}
		}
		pOptimizer->EpochStep(nEpoch + nInitEpoch);

		// Test phase
		pTestLdr->ResetCursor();
		pModel->TrainMode(false);
		float fTestLossSum = 0.f;
		std::vector<std::pair<int64_t, std::vector<float>>> testResults;
		for (uint64_t nIter = 1; pTestLdr->GetBatch(
				argBatchSize(), data, targets, device); ++nIter) {
			TENSOR_ARY outputs = pModel->Forward(std::move(data));
			auto iBeg = (nIter - 1) * argBatchSize();
			auto nNumRemains = argBatchSize();
			if (iBeg + argBatchSize() > pTestLdr->Size()) {
				nNumRemains = pTestLdr->Size() - iBeg;
				for (auto &out : outputs) {
					out = out.slice(0, 0, nNumRemains);
				}
				for (auto &t: targets) {
					t = t.slice(0, 0, nNumRemains);
				}
			}
			float fLoss = pLoss->Evaluate(std::move(outputs), std::move(targets));
			fTestLossSum += fLoss;
			if (argLogIters() > 0 && nIter % argLogIters() == 0) {
				LOG(INFO) << "test_iter=" << nIter
						  << ", loss=" << fLoss / nNumRemains;
			}
		}

		if (argEpochLog()) { // Logging phase
			std::string strTrainLoss = !bTrainMode ? "N/A"
					: std::to_string(fTrainLossSum / pTrainLdr->Size());
			std::string strTestLoss = !pTestLdr->Size() ? "N/A"
					: std::to_string(fTestLossSum / pTestLdr->Size());
			std::string strEpoch = !bTrainMode ? "TEST"
					: std::to_string(nEpoch + nInitEpoch);
			LOG(INFO) << "epoch " << strEpoch << ": train=" << strTrainLoss
					<< ", test=" << strTestLoss
					<< ", result: " << pLoss->FlushResults();
		}

		if (argStateSaveEpochs() != 0 && (nEpoch + nInitEpoch)
				% argStateSaveEpochs() == 0) {
			pModel->SaveWeights(MakeStateFilename(argStatePath(),
					argConfName(), nEpoch + nInitEpoch));
		}
	}
}
