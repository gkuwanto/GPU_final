#include "device_mine.cuh"
#include "../util/utils.hpp"
#include <iomanip>
#include <sstream>
#include <iostream>

using namespace std;

namespace sha2 {

	template <size_t N>
	using hash_array = std::array<uint8_t, N>;
	
	using sha256_hash = hash_array<32>;
	
	// SHA-2 uses big-endian integers.
	__device__ __host__ void write_u32(uint8_t* dest, uint32_t x)
	{
		*dest++ = (x >> 24) & 0xff;
		*dest++ = (x >> 16) & 0xff;
		*dest++ = (x >> 8) & 0xff;
		*dest++ = (x >> 0) & 0xff;
	}
	
	__device__ __host__ void write_u64(uint8_t* dest, uint64_t x)
	{
		*dest++ = (x >> 56) & 0xff;
		*dest++ = (x >> 48) & 0xff;
		*dest++ = (x >> 40) & 0xff;
		*dest++ = (x >> 32) & 0xff;
		*dest++ = (x >> 24) & 0xff;
		*dest++ = (x >> 16) & 0xff;
		*dest++ = (x >> 8) & 0xff;
		*dest++ = (x >> 0) & 0xff;
	}
	
	__device__ __host__  uint32_t read_u32(const uint8_t* src)
	{
		return static_cast<uint32_t>((src[0] << 24) | (src[1] << 16) |
									 (src[2] << 8) | src[3]);
	}
	
	__device__ __host__  uint64_t read_u64(const uint8_t* src)
	{
		uint64_t upper = read_u32(src);
		uint64_t lower = read_u32(src + 4);
		return ((upper & 0xffffffff) << 32) | (lower & 0xffffffff);
	}
	
	// A compiler-recognised implementation of rotate right that avoids the
	// undefined behaviour caused by shifting by the number of bits of the left-hand
	// type. See John Regehr's article https://blog.regehr.org/archives/1063
	__device__ __host__  uint32_t ror(uint32_t x, uint32_t n)
	{
		return (x >> n) | (x << (-n & 31));
	}
	
	__device__ __host__ uint64_t ror(uint64_t x, uint64_t n)
	{
		return (x >> n) | (x << (-n & 63));
	}
	
	// Both sha256_impl and sha512_impl are used by sha224/sha256 and
	// sha384/sha512 respectively, avoiding duplication as only the initial hash
	// values (s) and output hash length change.
	__device__ __host__ sha256_hash	sha256_impl(const uint32_t* s, const uint8_t* data, uint64_t length)
	{
		static_assert(sizeof(uint32_t) == 4, "sizeof(uint32_t) must be 4");
		static_assert(sizeof(uint64_t) == 8, "sizeof(uint64_t) must be 8");
	
		constexpr size_t chunk_bytes = 64;
		const uint64_t bit_length = length * 8;
	
		uint32_t hash[8] = {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]};
	
		constexpr uint32_t k[64] = {
			0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
			0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
			0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
			0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
			0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
			0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
			0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
			0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
			0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
			0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
			0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
	
		auto chunk = [=] __device__ __host__(const uint8_t* chunk_data, uint32_t* hash, uint32_t* k) {
			uint32_t w[64] = {0};
	
			for (int i = 0; i != 16; ++i) {
				w[i] = read_u32(&chunk_data[i * 4]);
			}
	
			for (int i = 16; i != 64; ++i) {
				auto w15 = w[i - 15];
				auto w2 = w[i - 2];
				auto s0 = ror(w15, 7) ^ ror(w15, 18) ^ (w15 >> 3);
				auto s1 = ror(w2, 17) ^ ror(w2, 19) ^ (w2 >> 10);
				w[i] = w[i - 16] + s0 + w[i - 7] + s1;
			}
	
			auto a = hash[0];
			auto b = hash[1];
			auto c = hash[2];
			auto d = hash[3];
			auto e = hash[4];
			auto f = hash[5];
			auto g = hash[6];
			auto h = hash[7];
	
			for (int i = 0; i != 64; ++i) {
				auto s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
				auto ch = (e & f) ^ (~e & g);
				auto temp1 = h + s1 + ch + k[i] + w[i];
				auto s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
				auto maj = (a & b) ^ (a & c) ^ (b & c);
				auto temp2 = s0 + maj;
	
				h = g;
				g = f;
				f = e;
				e = d + temp1;
				d = c;
				c = b;
				b = a;
				a = temp1 + temp2;
			}
	
			hash[0] += a;
			hash[1] += b;
			hash[2] += c;
			hash[3] += d;
			hash[4] += e;
			hash[5] += f;
			hash[6] += g;
			hash[7] += h;
		};
	
		while (length >= chunk_bytes) {
			chunk(data, (uint32_t *)hash,  (uint32_t *)k);
			data += chunk_bytes;
			length -= chunk_bytes;
		}
	
		{
			std::array<uint8_t, chunk_bytes> buf;
			memcpy(buf.data(), data, length);
	
			auto i = length;
			buf[i++] = 0x80;
	
			if (i > chunk_bytes - 8) {
				while (i < chunk_bytes) {
					buf[i++] = 0;
				}
	
				chunk(buf.data(), (uint32_t *) hash, (uint32_t *) k);
				i = 0;
			}
	
			while (i < chunk_bytes - 8) {
				buf[i++] = 0;
			}
	
			write_u64(&buf[i], bit_length);
	
			chunk(buf.data(), (uint32_t *) hash, (uint32_t *) k);
		}
	
		sha256_hash result;
	
		for (uint8_t i = 0; i != 8; ++i) {
			write_u32(&result[i * 4], hash[i]);
		}
	
		return result;
	}

	__device__ __host__ sha256_hash	sha256(const uint8_t* data, uint64_t length)
	{
		// First 32 bits of the fractional parts of the square roots of the first
		// eight primes 2..19:
		const uint32_t initial_hash_values[8] = {0x6a09e667,
												 0xbb67ae85,
												 0x3c6ef372,
												 0xa54ff53a,
												 0x510e527f,
												 0x9b05688c,
												 0x1f83d9ab,
												 0x5be0cd19};
	
		return sha256_impl(initial_hash_values, data, length);
	}
	
}


typedef struct {
	bool nonce_found;
	uint32_t nonce;
} Nonce_result;

uint32_t CPU_mine(const char* payload, uint32_t difficulty, uint32_t length) {
	for (uint32_t nonce = 0; nonce < 0xffffffff; nonce++){
		char data[length+8];
		for (int i = 0; i<length; i ++){
			data[i] = payload[i];
		}
		
		const char *a = "0123456789abcdef";

		data[length+0] = a[((nonce >> 28) % 16)];
		data[length+1] = a[((nonce >> 24) % 16)];
		data[length+2] = a[((nonce >> 20) % 16)];
		data[length+3] = a[((nonce >> 16) % 16)];
		data[length+4] = a[((nonce >> 12) % 16)];
		data[length+5] = a[((nonce >>  8) % 16)];
		data[length+6] = a[((nonce >>  4) % 16)];
		data[length+7] = a[((nonce) % 16)];
	
		auto ptr = reinterpret_cast<const uint8_t*>(data);
		sha2::sha256_hash hash = sha2::sha256(ptr, length+8);
		uint32_t i = 0;
		while (hash[i]==0){
			i++;
		}
		if(i>=difficulty){
			return nonce;
		}
	}
	return 0;

}

__global__ void GPU_mine(const char* payload, uint32_t difficulty, uint32_t length, uint32_t* result) {
	uint32_t nonce = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
	char data[160];
	for (int i = 0; i<length; i ++){
		data[i] = payload[i];
	}
	
	const char *a = "0123456789abcdef";

	data[length+0] = a[((nonce >> 28) % 16)];
	data[length+1] = a[((nonce >> 24) % 16)];
	data[length+2] = a[((nonce >> 20) % 16)];
	data[length+3] = a[((nonce >> 16) % 16)];
	data[length+4] = a[((nonce >> 12) % 16)];
	data[length+5] = a[((nonce >>  8) % 16)];
	data[length+6] = a[((nonce >>  4) % 16)];
	data[length+7] = a[((nonce) % 16)];

	auto ptr = reinterpret_cast<const uint8_t*>(data);
	// auto hash = sha2::sha256(ptr, length+8);
	uint32_t hash[32] = {0};
	uint32_t i = 0;
	while (hash[i]==0){
		i++;
	}
	if(i>=difficulty){
		*result = nonce;
	}
}

uint32_t device_mine_dispatcher(std::string payload, uint32_t difficulty, MineType reduction_type) {
    switch (reduction_type) {
        case MineType::MINE_CPU: {
            return CPU_mine(payload.c_str(), difficulty, payload.length());
        }

        case MineType::MINE_GPU: {
			const char *data = payload.c_str();
			uint32_t length = payload.length();
			uint32_t result = 0;

			uint32_t *dev_result;
			const char *dev_data;
			cudaMalloc((void **) &dev_data, (length+1) * sizeof(const char));
			cudaMalloc((void **) &dev_result, sizeof(uint32_t));
			cudaMemcpy(&dev_data, (void *) &data, (length+1) * sizeof(const char), cudaMemcpyHostToDevice);

			dim3 block(1024, 1);
			dim3 thread(512, 1);
			GPU_mine<<<block, thread >>>(dev_data, difficulty, length, dev_result);

			cudaDeviceSynchronize();


			cudaMemcpy((void *) &result, dev_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

            return result;
        }

    }
    return 0;
}