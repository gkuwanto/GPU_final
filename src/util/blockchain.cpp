#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include "blockchain.hpp"
#include "utils.hpp"
#include "sha256.hpp"



using namespace std;

CandidateBlock::CandidateBlock() {
    this->timestamp= std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
CandidateBlock::CandidateBlock(uint32_t difficulty) {
    this->timestamp= std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    this->difficulty = difficulty;
}
void CandidateBlock::setPreviousBlock(string prev_hash) {
    this->previous_block = prev_hash;
}
void CandidateBlock::setTransactionList(vector<string>& tx_list) {
    stringstream ss;
    for (vector<string>::iterator it = tx_list.begin(); it != tx_list.end(); it++) {
        ss << *it << "\n";
    }
    this->transaction_list_hash = hash_sha256(ss.str());
}
uint32_t CandidateBlock::getDifficulty(){
    return this->difficulty;
}

string CandidateBlock::getHashableString() {
    stringstream ss;
    ss << this->timestamp << this->previous_block<<this->transaction_list_hash;
    
    return ss.str();
}

Block::Block(CandidateBlock c_block, uint32_t nonce) {
    this->candidate_block = c_block;
    this->nonce = nonce;
}
string Block::getHashableString() {
    return this->candidate_block.getHashableString();
}

void change_non(SHA256_CTX *ctx, uint32_t nonce){
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
void Block::calculateHash(unsigned char *hash) {
    string payload = this->candidate_block.getHashableString();
    unsigned char *data = new unsigned char[payload.length() + 1];
            
	std::copy( payload.begin(), payload.end(), data );
	data[payload.length()] = 0;
	

	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char *) data, payload.length());	//ctx.state contains a-h

    change_non(&ctx, nonce);
    sha256_final(&ctx, hash);
    
}

bool Block::verify_nonce(){
    unsigned char hash[32];
    this->calculateHash(hash);
    uint32_t nDifficulty = this->candidate_block.getDifficulty();
    unsigned char difficulty[32];
    set_difficulty(difficulty, nDifficulty);
    int i=0;
    while(hash[i] == difficulty[i])
        i++;
    if(hash[i] < difficulty[i]){
        return true;
    }
    return false;
}

Blockchain::Blockchain() {
    this->current_difficulty = DEFAULT_DIFFICULTY;
}
uint32_t Blockchain::getDifficulty() {
    return this->current_difficulty;
}
void Blockchain::addBlock(Block new_block) {
    try {
        if (!new_block.verify_nonce()) {
            throw string("Block below difficulty level");
        }
        this->block_chain.push_back(new_block);
        
    } catch(string e) {
        cout << e << endl;
    }
}
string Blockchain::str() {
    stringstream ss;
    for (vector<Block>::iterator it = this->block_chain.begin(); it != this->block_chain.end(); it++) {
        ss << it->getHashableString() << "\n";
    }
    return ss.str();
}