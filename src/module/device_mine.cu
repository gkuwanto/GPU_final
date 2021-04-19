#include "device_mine.cuh"
#include "../util/utils.hpp"
#include <iomanip>
#include <sstream>
#include "sha256.cuh"
#include "sha256_unroll.h"

using namespace std;


inline void gpuAssert(cudaError_t code, char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

uint32_t CPU_mine(std::string payload, uint32_t difficulty) {
    for(uint32_t nonce = 0; nonce<0xffffffff; nonce++) {
        stringstream ss;
        ss << payload << setfill('0') << setw(8) << right << hex << nonce;
        string hash = hash_sha256(ss.str());
        if (hash.substr(0, difficulty) == string(difficulty, '0')) {
            return nonce;
        }
    }
    return 0;
}

typedef struct {
	bool nonce_found;
	uint32_t nonce;
} Nonce_result;
void initialize_nonce_result(Nonce_result *nr) {
	nr->nonce_found = false;
	nr->nonce = 0;
}

void GPU_mine(SHA256_CTX *ctx, Nonce_result *nr, ) {
    unsigned int m[64];
    unsigned int hash[8];
	unsigned int a,b,c,d,e,f,g,h,t1,t2;
	int i, j;
	uint32_t nonce = gridDim.x*blockDim.x + blockDim.x*blockIdx.x + threadIdx.x;
    sha256_change_nonce(ctx, nonce);
    unsigned int *le_data = (unsigned int *) ctx->data;
	for(i=0; i<16; i++)
		m[i] = le_data[i];
    
	for ( ; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    SHA256_COMPRESS_8X

	//Prepare input for next SHA-256
	m[0] = a + ctx->state[0];
	m[1] = b + ctx->state[1];
	m[2] = c + ctx->state[2];
	m[3] = d + ctx->state[3];
	m[4] = e + ctx->state[4];
	m[5] = f + ctx->state[5];
	m[6] = g + ctx->state[6];
	m[7] = h + ctx->state[7];
	//Pad the input
	m[8] = 0x80000000;	
	for(i=9; i<15; i++)
		m[i] = 0x00;
	m[15] = 0x00000100;	//Write out l=256
	for (i=16 ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	//Initialize the SHA-256 registers
	a = 0x6a09e667;
	b = 0xbb67ae85;
	c = 0x3c6ef372;
	d = 0xa54ff53a;
	e = 0x510e527f;
	f = 0x9b05688c;
	g = 0x1f83d9ab;
	h = 0x5be0cd19;

	SHA256_COMPRESS_1X

	hash[0] = ENDIAN_SWAP_32(a + 0x6a09e667);
	hash[1] = ENDIAN_SWAP_32(b + 0xbb67ae85);
	hash[2] = ENDIAN_SWAP_32(c + 0x3c6ef372);
	hash[3] = ENDIAN_SWAP_32(d + 0xa54ff53a);
	hash[4] = ENDIAN_SWAP_32(e + 0x510e527f);
	hash[5] = ENDIAN_SWAP_32(f + 0x9b05688c);
	hash[6] = ENDIAN_SWAP_32(g + 0x1f83d9ab);
	hash[7] = ENDIAN_SWAP_32(h + 0x5be0cd19);

    unsigned char *hhh = (unsigned char *) hash;
	i=0;
	while(hhh[i] == ctx->difficulty[i])
		i++;

	if(hhh[i] < ctx->difficulty[i]) {
		//Synchronization Issue
		//Kind of a hack but it really doesn't matter which nonce
		//is written to the output, they're all winners :)
		//Further it's unlikely to even find a nonce let alone 2
		nr->nonce_found = true;
		//The nonce here has the correct endianess,
		//but it must be stored in the block in little endian order
		nr->nonce = nonce;
	}
}

uint32_t device_mine_dispatcher(std::string payload, uint32_t difficulty, MineType reduction_type) {
    switch (reduction_type) {
        case MineType::MINE_CPU: {
            return CPU_mine(payload, difficulty);
        }

        default: {
            unsigned char *data = new unsigned char[payload.length() + 1];
            
            std::copy( payload.begin(), payload.end(), data );
            data[payload.length()] = 0;
            

            Nonce_result h_nr;
            initialize_nonce_result(&h_nr);
            
            SHA256_CTX ctx;
            sha256_init(&ctx);
            sha256_update(&ctx, (unsigned char *) data, payload.length());	//ctx.state contains a-h
            set_difficulty(ctx.difficulty, difficulty);


            SHA256_CTX *d_ctx;
            Nonce_result *d_nr;
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_ctx, sizeof(SHA256_CTX)));
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));
            CUDA_SAFE_CALL(cudaMemcpy(d_ctx, (void *) &ctx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &h_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

            // 4194304 * 1024 = 0xffffffff + 1
            GPU_mine<<<4194304, 1024>>>(d_ctx, d_nr);


            CUDA_SAFE_CALL(cudaMemcpy((void *) &h_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

        }

    }
    return 0;
}