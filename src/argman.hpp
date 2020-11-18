#ifndef __ARGMAN_HPP
#define __ARGMAN_HPP

#include <functional>
#include <string>
#include <vector>
#include "json.hpp"

class ArgMan {
public:
	using BASE_PARSER = std::function<void(const nlohmann::json &jValue)>;
	void Register(std::string strName, BASE_PARSER parser);
	std::map<std::string, BASE_PARSER>& GetRegMap();
private:
	std::map<std::string, BASE_PARSER> m_RegMap;
};

extern ArgMan g_MainArgs;

template<typename _Ty>
void Parser(const nlohmann::json &jValue, _Ty &val) {
	try {
		val = jValue.get<_Ty>();
	} catch (...) {
	}
}

template<typename _Ty>
class Arg {
public:
	using PARSER = std::function<void(const nlohmann::json &jValue, _Ty &val)>;
	Arg(std::string strName, const _Ty &defVal, PARSER parser,
			ArgMan &argman = g_MainArgs)
			: m_Parser(std::move(parser)) {
		m_Var = defVal;
		argman.Register(std::move(strName), std::bind(m_Parser,
				std::placeholders::_1, std::ref(m_Var)));
	}
	Arg(std::string strName, PARSER parser,
			ArgMan &argman = g_MainArgs)
			: m_Parser(std::move(parser)) {
		argman.Register(std::move(strName), std::bind(m_Parser,
				std::placeholders::_1, std::ref(m_Var)));
	}
	Arg(std::string strName, const _Ty &defVal, ArgMan &argman = g_MainArgs)
			: m_Parser(Parser<_Ty>) {
		m_Var = defVal;
		argman.Register(std::move(strName), std::bind(m_Parser,
				std::placeholders::_1, std::ref(m_Var)));
	}
	Arg(std::string strName, ArgMan &argman = g_MainArgs)
			: m_Parser(std::move(Parser<_Ty>)) {
		argman.Register(std::move(strName), std::bind(m_Parser,
				std::placeholders::_1, std::ref(m_Var)));
	}
	void Set(const _Ty &val) {
		m_Var = val;
	}
	_Ty& operator()() {
		return m_Var;
	}
	const _Ty& operator()() const {
		return m_Var;
	}
private:
	_Ty m_Var;
	PARSER m_Parser;
};

void ParseArgsFromJson(const nlohmann::json &jCfg, ArgMan &argman = g_MainArgs);

std::string PathAddSlash(const std::string &strPath);

#endif // __ARGMAN_HPP
