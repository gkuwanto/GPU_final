#include "device_mine.cuh"
#include "../util/utils.hpp"
#include <iomanip>
#include <sstream>
#include <iostream>
#include "../util/sha256.hpp"
#include "../util/sha256_unroll.hpp"

using namespace std;

#define ENDIAN_SWAP_32(x) (\
	((x & 0xff000000) >> 24) | \
	((x & 0x00ff0000) >> 8 ) | \
	((x & 0x0000ff00) << 8 ) | \
	((x & 0x000000ff) << 24))


__constant__ uint32_t k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }



typedef struct {
	bool nonce_found;
	uint32_t nonce;
} Nonce_result;

void initialize_nonce_result(Nonce_result *nr) {
	nr->nonce_found = false;
	nr->nonce = 0;
}


__device__ __host__ void sha256_change_nonce(SHA256_CTX *ctx, uint32_t nonce)
{
	char *b = "0123456789abcdef";
	unsigned char *a = (unsigned char *)b;
	BYTE data[8] = {
		a[((nonce >> 28) % 16)], a[((nonce >> 24) % 16)], a[((nonce >> 20) % 16)], 
		a[((nonce >> 16) % 16)], a[((nonce >> 12) % 16)], a[((nonce >> 8)  % 16)],
		a[((nonce >> 4)  % 16)], a[(nonce % 16)]
	};

	WORD i;

	for (i = 0; i < 8; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

			for (i = 0, j = 0; i < 16; ++i, j += 4)
				m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
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

			for (i = 0; i < 64; ++i) {
				t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
				t2 = EP0(a) + MAJ(a,b,c);
				h = g;
				g = f;
				f = e;
				e = d + t1;
				d = c;
				c = b;
				b = a;
				a = t1 + t2;
			}

			ctx->state[0] += a;
			ctx->state[1] += b;
			ctx->state[2] += c;
			ctx->state[3] += d;
			ctx->state[4] += e;
			ctx->state[5] += f;
			ctx->state[6] += g;
			ctx->state[7] += h;
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}



	
}


uint32_t CPU_mine(std::string payload, uint32_t difficulty) {

	unsigned char *data = new unsigned char[payload.length() + 1];
            
	std::copy( payload.begin(), payload.end(), data );
	data[payload.length()] = 0;
	
	int i, j;

	Nonce_result h_nr;
	initialize_nonce_result(&h_nr);
	
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char *) data, payload.length());	//ctx.state contains a-h
	set_difficulty(ctx.difficulty, difficulty);

	unsigned int *le_data = (unsigned int *)ctx.data;
	unsigned int le;
	for(i=0, j=0; i<16; i++, j+=4) {
		//Get the data out as big endian
		//Store it as little endian via x86
		//On the device side cast the pointer as int* and dereference it correctly
		le = (ctx.data[j] << 24) | (ctx.data[j + 1] << 16) | (ctx.data[j + 2] << 8) | (ctx.data[j + 3]);
		le_data[i] = le;
	}

    for(uint32_t nonce = 0; nonce<0xffffffff; nonce++) {
        unsigned int m[64];
		unsigned int hash[8];
		unsigned int a,b,c,d,e,f,g,h,t1,t2;
		sha256_change_nonce(&ctx, nonce);
		unsigned int *le_data_loop = (unsigned int *) (&ctx)->data;
		for(i=0; i<16; i++)
			m[i] = le_data_loop[i];
		
		for ( ; i < 64; ++i)
			m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

		a = (&ctx)->state[0];
		b = (&ctx)->state[1];
		c = (&ctx)->state[2];
		d = (&ctx)->state[3];
		e = (&ctx)->state[4];
		f = (&ctx)->state[5];
		g = (&ctx)->state[6];
		h = (&ctx)->state[7];

		SHA256_COMPRESS_8X

		//Prepare input for next SHA-256
		m[0] = a + (&ctx)->state[0];
		m[1] = b + (&ctx)->state[1];
		m[2] = c + (&ctx)->state[2];
		m[3] = d + (&ctx)->state[3];
		m[4] = e + (&ctx)->state[4];
		m[5] = f + (&ctx)->state[5];
		m[6] = g + (&ctx)->state[6];
		m[7] = h + (&ctx)->state[7];
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
		cout<<*hash<<endl;

		i=0;
		while(hhh[i] == (&ctx)->difficulty[i])
			i++;
		cout<<i;
		if(hhh[i] < (&ctx)->difficulty[i]) {
			//Synchronization Issue
			//Kind of a hack but it really doesn't matter which nonce
			//is written to the output, they're all winners :)
			//Further it's unlikely to even find a nonce let alone 2
			(&h_nr)->nonce_found = true;
			//The nonce here has the correct endianess,
			//but it must be stored in the block in little endian order
			(&h_nr)->nonce = nonce;
			return h_nr.nonce;
		}
		break;
    }
    return 0;
}

__global__ void GPU_mine(SHA256_CTX *ctx, Nonce_result *nr ) {
	const uint32_t k[64] = {
		0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
		0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
		0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
		0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
		0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
		0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
		0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
		0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
	};
    unsigned int m[64];
    unsigned int hash[8];
	unsigned int a,b,c,d,e,f,g,h,t1,t2;
	int i, j;
	uint32_t nonce = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
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

        case MineType::MINE_GPU: {
            unsigned char *data = new unsigned char[payload.length() + 1];
            
            std::copy( payload.begin(), payload.end(), data );
            data[payload.length()] = 0;
            

            Nonce_result h_nr;
            initialize_nonce_result(&h_nr);
            
            SHA256_CTX ctx;
            sha256_init(&ctx);
            sha256_update(&ctx, (unsigned char *) data, payload.length());	//ctx.state contains a-h
            set_difficulty(ctx.difficulty, difficulty);
			unsigned int *le_data = (unsigned int *)ctx.data;
			unsigned int le;
			int i, j;
			for(i=0, j=0; i<16; i++, j+=4) {
				//Get the data out as big endian
				//Store it as little endian via x86
				//On the device side cast the pointer as int* and dereference it correctly
				le = (ctx.data[j] << 24) | (ctx.data[j + 1] << 16) | (ctx.data[j + 2] << 8) | (ctx.data[j + 3]);
				le_data[i] = le;
			}


            SHA256_CTX *d_ctx;
            Nonce_result *d_nr;
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_ctx, sizeof(SHA256_CTX)));
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));
            CUDA_SAFE_CALL(cudaMemcpy(d_ctx, (void *) &ctx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &h_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

            // 4194304 * 1024 = 0xffffffff + 1

			dim3 gridDim(8192,8192);
			// dim3 gridDim(1,1);

			dim3 blockDim(64,1);
            GPU_mine<<<gridDim, blockDim>>>(d_ctx, d_nr);

			cudaDeviceSynchronize();

            CUDA_SAFE_CALL(cudaMemcpy((void *) &h_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

			cudaDeviceSynchronize();

			cout << h_nr.nonce_found << h_nr.nonce;
			return h_nr.nonce;
        }

    }
    return 0;
}