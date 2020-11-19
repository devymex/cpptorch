#ifndef __CREATOR_HPP
#define __CREATOR_HPP

#include <map>
#include <memory>
#include <functional>
#include <glog/logging.h>
#include "json.hpp"

template<typename _BASE>
class Creator {
public:
	using BASE_PTR = std::unique_ptr<_BASE>;
	using CREATOR_FUNC = std::function<BASE_PTR()>;
	using CREATOR_MAP = std::map<std::string, CREATOR_FUNC>;

	static BASE_PTR Create(const nlohmann::json &jConf) {
		auto iType = jConf.find("name");
		CHECK(iType != jConf.end());
		CHECK(iType->is_string());
		std::string strName = iType->get<std::string>();
		CHECK(!strName.empty());
		auto iCreator = __GetCreatorMap().find(strName);
		CHECK(iCreator != __GetCreatorMap().end())
				<< "Invalid name: " << strName;
		BASE_PTR pObj = iCreator->second();
		pObj->Initialize(jConf);
		return pObj;
	}

	static void RegisterCreator(const std::string &strName,
			CREATOR_FUNC creator) {
		CHECK(!strName.empty()) << "Engine name is empty!";
		CHECK(__GetCreatorMap().find(strName) == __GetCreatorMap().end())
				<< "Name already exists: " << strName;
		__GetCreatorMap()[strName] = std::move(creator);
	}

private:
	static CREATOR_MAP& __GetCreatorMap() {
		static std::unique_ptr<CREATOR_MAP> pCreatorMap;
		if (pCreatorMap == nullptr) {
			pCreatorMap.reset(new CREATOR_MAP);
		}
		return *pCreatorMap;
	}
};

template<typename _BASE, typename _CLASS>
class CreatorRegister {
public:
	CreatorRegister(const std::string &strName) {
		typename Creator<_BASE>::CREATOR_FUNC creatorFunc = [=] {
			return std::unique_ptr<_BASE>(new _CLASS);
		};
		Creator<_BASE>::RegisterCreator(strName, creatorFunc);
	}
};

#define REGISTER_CREATOR(base, _class, name) \
	CreatorRegister<base, _class> g_RegHelper_##_class(name);

#endif // #ifndef __DATA_LOADER_HPP