#ifndef __DEVICE_MINE_CUH__
#define __DEVICE_MINE_CUH__

#include <string>
enum MineType { MINE_CPU, MINE_NAIVE };

uint32_t CPU_mine(std::string, uint32_t);
uint32_t device_mine_dispatcher(std::string, uint32_t, MineType);

#endif