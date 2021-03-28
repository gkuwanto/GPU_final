#ifndef __DEVICE_VERIFY_CUH__
#define __DEVICE_VERIFY_CUH__

#include <map>
#include <string>
#include "../util/utils.hpp"

enum VerifyType { CPU };

void device_verify_dispatcher(std::map<std::string, Transaction>&, VerifyType);
bool CPU_verify(std::map<std::string, Transaction>&);

#endif