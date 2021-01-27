/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* CTimer for Code Profile
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
*/

#ifndef CTIMER_HPP_
#define CTIMER_HPP_

#include <chrono>
#include <map>
#include <sstream>
#include <string>

class CTimer
{
protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
	std::map<int, double> m_Records;
public:
	CTimer() {
		Start();
	}

	inline void Start() {
		m_start = std::chrono::high_resolution_clock::now();
	}

	inline double Now() {
		auto now = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = now - m_start;
		return diff.count();
	}

	inline double Reset(int nCheckPoint = -1) {
		double dNow = Now();
		if (nCheckPoint >= 0) {
			m_Records[nCheckPoint] += dNow;
		}
		Start();
		return dNow;
	}

	inline void Clear() {
		m_Records.clear();
	}

	std::string Format() {
		std::ostringstream oss;
		auto r = m_Records.begin();
		if (r != m_Records.end()) {
			oss << "CP" << r->first << "=" << r->second ;
		}
		for (++r; r != m_Records.end(); ++r) {
			oss << ", CP" << r->first << "=" << r->second;
		}
		return oss.str();
	}
};

#endif // CTIMER_HPP_
