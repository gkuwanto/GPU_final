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


// sha256 function device

__device__ void d_set_difficulty(unsigned char *difficulty, unsigned int nBits) {
	unsigned int i;
	for(i=0; i<32; i++) {
		if (i < nBits) {
			difficulty[i] = 0;
		}
		else {
			difficulty[i] = 'f';
		}
	}
	difficulty[31] = 0;
	
}


/*********************** FUNCTION DEFINITIONS ***********************/
__device__ void d_sha256_transform(SHA256_CTX *ctx, const BYTE data[])
{
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
}

__device__ void d_sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void d_sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void d_sha256_final(SHA256_CTX *ctx, BYTE hash[])
{
	WORD i;

	sha256_pad(ctx);
	sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

__device__ void d_sha256_pad(SHA256_CTX *ctx)
{
	WORD i;

	//How many bytes exist in the remainder buffer
	i = ctx->datalen;

	//Pad whatever data is left in the buffer.
	//If it's less than 56 bytes, 8 bytes required for bit length l
	//For this application we are always the first case
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	//Otherwise, pad with 0s and store l in its own message block
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	//Store value of l
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
}

//end




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
	#ifdef  __CUDA_ARCH__

	#else

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

	#endif

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
	
	int i;

	Nonce_result h_nr;
	initialize_nonce_result(&h_nr);
	
	

    for(uint32_t nonce = 0; nonce<0xffffffff; nonce++) {
        SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char *) data, payload.length());	//ctx.state contains a-h
	set_difficulty(ctx.difficulty, difficulty);
		unsigned char hash[32];
		sha256_change_nonce(&ctx, nonce);
		sha256_final(&ctx, hash);

		i=0;
		while(hash[i] == (&ctx)->difficulty[i])
			i++;
		if(hash[i] < (&ctx)->difficulty[i]) {
			(&h_nr)->nonce_found = true;
			(&h_nr)->nonce = nonce;
			cout << hash[0] << hash[1];
			return h_nr.nonce;
		}
    }
    return 0;
}

__global__ void GPU_mine(unsigned char *data, Nonce_result *nr , size_t length, uint32_t difficulty) {

	SHA256_CTX ctx;
	d_sha256_init(&ctx);
	d_sha256_update(&ctx, (unsigned char *) data, length-1);	//ctx.state contains a-h
	d_set_difficulty(ctx.difficulty, difficulty);

	uint32_t nonce = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
    sha256_change_nonce(&ctx, nonce);
	unsigned char hash[32];
	
	// pad
	WORD i;
	i = (&ctx)->datalen;

	//Pad whatever data is left in the buffer.
	//If it's less than 56 bytes, 8 bytes required for bit length l
	//For this application we are always the first case
	if ((&ctx)->datalen < 56) {
		(&ctx)->data[i++] = 0x80;
		while (i < 56)
			(&ctx)->data[i++] = 0x00;
	}
	//Otherwise, pad with 0s and store l in its own message block
	else {
		(&ctx)->data[i++] = 0x80;
		while (i < 64)
			(&ctx)->data[i++] = 0x00;
		// transform
		WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

		for (i = 0, j = 0; i < 16; ++i, j += 4)
			m[i] = ((&ctx)->data[j] << 24) | ((&ctx)->data[j + 1] << 16) | ((&ctx)->data[j + 2] << 8) | ((&ctx)->data[j + 3]);
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

		(&ctx)->state[0] += a;
		(&ctx)->state[1] += b;
		(&ctx)->state[2] += c;
		(&ctx)->state[3] += d;
		(&ctx)->state[4] += e;
		(&ctx)->state[5] += f;
		(&ctx)->state[6] += g;
		(&ctx)->state[7] += h;

		// end transform
		memset((&ctx)->data, 0, 56);
	}

	//Store value of l
	(&ctx)->bitlen += (&ctx)->datalen * 8;
	(&ctx)->data[63] = (&ctx)->bitlen;
	(&ctx)->data[62] = (&ctx)->bitlen >> 8;
	(&ctx)->data[61] = (&ctx)->bitlen >> 16;
	(&ctx)->data[60] = (&ctx)->bitlen >> 24;
	(&ctx)->data[59] = (&ctx)->bitlen >> 32;
	(&ctx)->data[58] = (&ctx)->bitlen >> 40;
	(&ctx)->data[57] = (&ctx)->bitlen >> 48;
	(&ctx)->data[56] = (&ctx)->bitlen >> 56;

	// transform

	WORD a, b, c, d, e, f, g, h, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = ((&ctx)->data[j] << 24) | ((&ctx)->data[j + 1] << 16) | ((&ctx)->data[j + 2] << 8) | ((&ctx)->data[j + 3]);
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

	(&ctx)->state[0] += a;
	(&ctx)->state[1] += b;
	(&ctx)->state[2] += c;
	(&ctx)->state[3] += d;
	(&ctx)->state[4] += e;
	(&ctx)->state[5] += f;
	(&ctx)->state[6] += g;
	(&ctx)->state[7] += h;

	// end transform

	for (i = 0; i < 4; ++i) {
		hash[i]      = ((&ctx)->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = ((&ctx)->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = ((&ctx)->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = ((&ctx)->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = ((&ctx)->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = ((&ctx)->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = ((&ctx)->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = ((&ctx)->state[7] >> (24 - i * 8)) & 0x000000ff;
	}

	// Check hash
	i=0;
	while(hash[i] == (&ctx)->difficulty[i])
		i++;
	if(hash[i] < (&ctx)->difficulty[i]) {
		nr->nonce_found = true;
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
			size_t length = payload.length()+1;
            
            std::copy( payload.begin(), payload.end(), data );
            data[payload.length()] = 0;
            

            Nonce_result h_nr;
            initialize_nonce_result(&h_nr);
            
			unsigned char *d_data;

            Nonce_result *d_nr;
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, length * sizeof(unsigned char)));
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));
            CUDA_SAFE_CALL(cudaMemcpy(d_data, (void *) &data, length * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &h_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

            // 8192 * 8192 * 64 = 0xffffffff + 1

			// dim3 gridDim(8192,8192);
			dim3 gridDim(1,1);

			dim3 blockDim(64,1);
            GPU_mine<<<gridDim, blockDim>>>(d_data, d_nr, length, difficulty);

			cudaDeviceSynchronize();

            CUDA_SAFE_CALL(cudaMemcpy((void *) &h_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

			cudaDeviceSynchronize();

			return h_nr.nonce;
        }

    }
    return 0;
}