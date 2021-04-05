#ifndef __DEVICE_VERIFY_CUH__
#define __DEVICE_VERIFY_CUH__

#include <string>
enum MineType { CPU, NAIVE };

uint32_t CPU_mine(std::string, uint32_t);
uint32_t device_mine_dispatcher(std::string, uint32_t, MineType);

#endif