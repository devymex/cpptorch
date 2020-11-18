#include <ctime>
#include <fstream>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "data_loader.hpp"
#include "weights.hpp"
#include "model.hpp"
#include "utils.hpp"
#include "argman.hpp"
#include "json.hpp"

namespace bfs = boost::filesystem;
namespace tfunc = torch::nn::functional;

std::pair<uint32_t, std::string> FindLastState(const std::string &strPath,
		 const std::string &strName) {
	std::string strPrefix = "^" + strName + "_";
	std::string strNumber = "e([1-9][0-9]*)";
	std::string strSuffix = "\\.pkl\\.2[0-9]{11}$";
	std::string strPattern = strPrefix + strNumber + strSuffix;
	std::pair<uint32_t, std::string> ret;
	auto filenames = EnumerateFiles(strPath, strPattern);
	if (!filenames.empty()) {
		std::vector<size_t> epochs;
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
		const std::string &strModelName, uint32_t nEpoch) {
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
	Arg<size_t> argDeviceID("device_id", 0);
	Arg<std::string> argDataRoot("data_root");
	Arg<std::string> argTrainList("train_list");
	Arg<std::string> argTestList("test_list");
	Arg<std::string> argModelFile("model_file");
	Arg<bool> argLabelBalance("label_balance", true);
	Arg<bool> argReinitWeights("reinit_weights", false);
	Arg<int32_t> argRandomSeed("random_seed", -1);
	Arg<size_t> argMaxEpoch("max_epoch", 10);
	Arg<size_t> argBatchSize("batch_size", 32);
	Arg<float> argLearningRate("learning_rate", 0.01f);
	Arg<size_t> argLRStepSize("lr_step_epochs", 0);
	Arg<float> argLRStepGamma("lr_step_gamma", 0.f);
	Arg<float> argWeightDecay("weight_decay", 0.f);
	Arg<float> argMomentum("momentum", 0.f);
	Arg<std::string> argLogPath("log_path");
	Arg<size_t> argLogIters("log_iters", 0);
	Arg<std::string> argStatePath("state_path");
	Arg<size_t> argStateSaveEpochs("save_state_epochs", 0);
	Arg<bool> argLoadLast("load_last", false);
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
	auto pTrainLdr = CreateDataLoader(argTrainData());
	auto pTestLdr = CreateDataLoader(argTestData());

	// Module Preparation
	// -------------------------------------------------------------------------
	size_t nInitEpoch = 0;
	Net pModel;
	pModel->to(device);
	if (argLoadLast()) {
		auto lastState = FindLastState(argStatePath(), argConfName());
		nInitEpoch = lastState.first;
		pModel->load_weights(argModelFile());
		if (nInitEpoch > 0) {
			LOG(INFO) << "Weights loaded from last epoch " << nInitEpoch;
		}
	} else if (argReinitWeights()) {
		pModel->initialize(InitModuleWeight);
		LOG(INFO) << "Weights reinitialized";
	}
	if (!bTrainMode) {
		argMaxEpoch.Set(nInitEpoch + 1);
		CHECK(!argTestResultsFile().empty());
	}

	// Optimizer Initialization
	// -------------------------------------------------------------------------
	std::vector<torch::Tensor> parameters;
	for (const auto &pair : pModel->named_parameters()) {
		parameters.push_back(pair.second);
	}
	float fLearningRate = argLearningRate();
	if (argLRStepSize() > 0) {
		uint32_t nSteps = nInitEpoch / argLRStepSize();
		fLearningRate *= std::pow(argLRStepGamma(), (float)nSteps);
	}
	using SGDOPTION = torch::optim::SGDOptions;
	torch::optim::SGD sgdOptim(parameters, SGDOPTION(fLearningRate)
			.weight_decay(argWeightDecay())
			.momentum(argMomentum()));

	// Main Loop
	// -------------------------------------------------------------------------
	for (uint32_t nEpoch = 1; nEpoch + nInitEpoch <= argMaxEpoch(); ++nEpoch) {
		torch::Tensor tData, tTarget;
		auto &sgdOption = static_cast<SGDOPTION&>(sgdOptim.defaults());
		// Train phase
		if (argLRStepSize() > 0 && (nEpoch + nInitEpoch) % argLRStepSize() == 0) {
			uint32_t nSteps = (nEpoch + nInitEpoch) / argLRStepSize();
			float fDecay = std::pow(argLRStepGamma(), (float)nSteps);
			sgdOption.lr(argLearningRate() * fDecay);
		}
		pTrainLdr->ResetCursor();
		pModel->train();
		float fTrainLoss = 0.f;
		for (uint32_t nIter = 1; bTrainMode && pTrainLdr->GetBatch(
				argBatchSize(), device, tData, tTarget); ++nIter) {
			torch::Tensor tOutput = pModel->forward({tData}).toTensor();
			torch::Tensor tSoftmax = tfunc::log_softmax(tOutput, 1);
			torch::Tensor tLoss = tfunc::nll_loss(tSoftmax, tTarget);
			fTrainLoss += tLoss.item().toFloat() * argBatchSize();
			sgdOptim.zero_grad();
			tLoss.backward();
			sgdOptim.step();
			if (argLogIters() > 0 && nIter % argLogIters() == 0) {
				LOG(INFO) << "train_iter=" << nIter << ", loss="
						  << tLoss.item().toFloat();
			}
		}
		// Test phase
		int nCorrectCnt = 0;
		pTestLdr->ResetCursor();
		pModel->train(false);
		float fTestLoss = 0.f;
		std::vector<std::pair<int64_t, std::vector<float>>> testResults;
		for (uint32_t nIter = 1; pTestLdr->GetBatch(
				argBatchSize(), device, tData, tTarget); ++nIter) {
			torch::Tensor tOutput = pModel->forward({tData}).toTensor();
			torch::Tensor tSoftmax = tfunc::log_softmax(tOutput, 1);
			torch::Tensor tLoss = tfunc::nll_loss(tSoftmax, tTarget,
					tfunc::NLLLossFuncOptions().reduction(torch::kSum));
			fTestLoss += tLoss.item().toFloat();
			torch::Tensor tPred = tOutput.argmax(1, true);

			tTarget = tTarget.to(torch::kCPU).flatten().contiguous();
			tPred = tPred.to(torch::kCPU).flatten().contiguous();
			tSoftmax = tSoftmax.to(torch::kCPU).contiguous();
			size_t nClassNum = tSoftmax.size(1);
			for (size_t i = 0; i < argBatchSize(); ++i) {
				uint32_t nIdx = (nIter - 1) * argBatchSize() + i;
				if (nIdx < pTestLdr->Size()) {
					int64_t nPredLabel = tPred[i].item().toLong();
					nCorrectCnt += (nPredLabel == tTarget[i].item().toLong());
					if (!bTrainMode) {
						float *pSoftmax = tSoftmax[i].data_ptr<float>();
						testResults.emplace_back(nPredLabel, std::vector<float>(
								pSoftmax, pSoftmax + nClassNum));
					}
				}
			}
			if (argLogIters() > 0 && nIter % argLogIters() == 0) {
				LOG(INFO) << "test_iter=" << nIter << ", loss="
						  << tLoss.item().toFloat();
			}
		}
		// Logging phase
		if (!bTrainMode) {
			std::ofstream testResFile(argTestResultsFile());
			CHECK(testResFile.is_open());
			CHECK_EQ(testResults.size(), pTestLdr->Size());
			for (auto &row: testResults) {
				testResFile << row.first;
				for (auto v: row.second) {
					testResFile << " " << v;
				}
				testResFile << std::endl;
			}
			LOG(INFO) << "Tesing results saved into \""
					  << argTestResultsFile() << "\"";
		}
		std::string strTrainLoss = !bTrainMode ? "N/A"
				: std::to_string(fTrainLoss / pTrainLdr->Size());
		std::string strEpoch = !bTrainMode ? "TEST"
				: std::to_string(nEpoch + nInitEpoch);
		LOG(INFO) << "epoch " << strEpoch << ": lr=" << sgdOption.lr()
				  << ", train=" << strTrainLoss
				  << ", test=" << fTestLoss / pTestLdr->Size()
				  << ", acc=" << (float)nCorrectCnt / pTestLdr->Size();
		if (argStateSaveEpochs() != 0 && (nEpoch + nInitEpoch)
				% argStateSaveEpochs() == 0) {
			pModel->save_weights(MakeStateFilename(argStatePath(),
					argConfName(), nEpoch + nInitEpoch));
		}
	}
}
