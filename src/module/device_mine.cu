#include "device_mine.cuh"
#include "../util/utils.hpp"
#include <iomanip>
#include <sstream>

using namespace std;

uint32_t CPU_mine(std::string payload, uint32_t difficulty) {
    for(uint32_t nonce = 0; nonce<0xffffffff; nonce++) {
        stringstream ss;
        ss << payload << hex << nonce;
        string hash = hash_sha256(ss.str());
        if (hash.substr(0, difficulty) == string(difficulty, '0')) {
            return nonce;
        }
    }
    return -1;
}

uint32_t device_mine_dispatcher(std::string payload, uint32_t difficulty, MineType reduction_type) {
    switch (reduction_type) {
        case MineType::CPU: {
            return CPU_mine(payload, difficulty);
        }
    }
    return -1;
}`