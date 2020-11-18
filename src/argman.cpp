#include <glog/logging.h>
#include "argman.hpp"

ArgMan g_MainArgs;

void ArgMan::Register(std::string strName, ArgMan::BASE_PARSER parser) {
	auto &regMap = GetRegMap();
	CHECK_EQ(regMap.count(strName), 0);
	regMap[std::move(strName)] = std::move(parser);
}

std::map<std::string, ArgMan::BASE_PARSER>& ArgMan::GetRegMap() {
	return m_RegMap;
}

void ParseArgsFromJson(const nlohmann::json &jCfg, ArgMan &argman) {
	auto &regMap = argman.GetRegMap();
	for (auto &jArg : jCfg.items()) {
		std::string strKey = jArg.key();
		auto iParser = regMap.find(jArg.key());
		if (iParser != regMap.end()) {
			iParser->second(jArg.value());
		}
	}
}

template<>
void Parser(const nlohmann::json &jValue, size_t &val) {
	CHECK(jValue.is_number_unsigned());
	val = jValue.get<size_t>();
}

template<>
void Parser(const nlohmann::json &jValue, int32_t &val) {
	CHECK(jValue.is_number_integer());
	val = jValue.get<int32_t>();
}

template<>
void Parser(const nlohmann::json &jValue, float &val) {
	CHECK(jValue.is_number_float());
		val = jValue.get<float>();
}

template<>
void Parser(const nlohmann::json &jValue, bool &val) {
	CHECK(jValue.is_boolean());
	val = jValue.get<bool>();
}

template<>
void Parser(const nlohmann::json &jValue, std::string &val) {
	CHECK(jValue.is_string());
	val = jValue.get<std::string>();
}

std::string PathAddSlash(const std::string &strPath) {
	if (!strPath.empty() && strPath.back() != '/') {
		return strPath + "/";
	}
	return strPath;
}

