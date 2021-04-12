#include "device_mine.cuh"
#include "../util/utils.hpp"
#include <iomanip>
#include <sstream>

using namespace std;

uint32_t CPU_mine(string payload, uint32_t difficulty) {
    for(uint32_t nonce = 0; nonce<0xffffffff; nonce++) {
        stringstream ss;
        ss << payload << hex << nonce;
        string hash = hash_sha256(ss.str());
        if (hash.substr(0, difficulty) == string(difficulty, '0')) {
            return nonce;
        }
    }
    return 0;
}

__global__ void GPU_naive_mine(char *payload, uint32_t *difficulty, uint32_t* nonce, bool* nonce_found) {
    if (!nonce_found[0]){
        uint32_t candidate_nonce = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
        stringstream ss;
        ss << payload << hex << candidate_nonce;
        string hash = hash_sha256(ss.str());
        if (hash.substr(0, *difficulty) == string(*difficulty, '0')){
            nonce[0] = candidate_nonce;
            nonce_found[0] = true;
        }
    }
}



uint32_t device_mine_dispatcher(string payload, uint32_t difficulty, MineType reduction_type) {
    switch (reduction_type) {
        case MineType::MINE_CPU: {
            return CPU_mine(payload, difficulty);
        }
        case MineType::MINE_NAIVE: {
            uint32_t *nonce;
            bool *nonce_found;
            char const * c_payload;
            uint32_t *dev_dif;
            cudaMallocManaged(&nonce, sizeof(uint32_t));
            cudaMallocManaged(&dev_dif, sizeof(uint32_t));
            cudaMallocManaged(&nonce_found, sizeof(bool));
            cudaMallocManaged(&c_payload, payload.length()*sizeof(char));
            nonce_found[0] = false;
            c_payload = payload.c_str();
            *dev_dif = difficulty;
            int blockSize = 1024;
            int numBlocks = 2; //4194304; // ceil(0xffffffff/1024)
            GPU_naive_mine<<numBlocks, blockSize>>(c_payload, dev_dif, nonce, nonce_found);
            cudaDeviceSynchronize();
            return nonce[0];
        }
    }
    return 0;
}