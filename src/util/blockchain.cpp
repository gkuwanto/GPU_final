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
    this->transaction_list = tx_list;
}
uint32_t CandidateBlock::getDifficulty(){
    return this->difficulty;
}

string CandidateBlock::getHashableString() {
    stringstream ss;
    ss << this->timestamp << this->previous_block;
    for (vector<string>::iterator it = this->transaction_list.begin(); it != this->transaction_list.end(); it++) {
        ss << *it << "\n";
    }
    return ss.str();
}

Block::Block(CandidateBlock c_block, uint32_t nonce) {
    this->candidate_block = c_block;
    this->nonce = nonce;
}
string Block::getHashableString() {
    return this->candidate_block.getHashableString();
}
void Block::calculateHash(unsigned char *hash) {
    string payload = this->candidate_block.getHashableString();

    char *b = "0123456789abcdef";
	unsigned char *a = (unsigned char *)b;
	unsigned char nonce_byte[8] = {
		a[((nonce >> 28) % 16)], a[((nonce >> 24) % 16)], a[((nonce >> 20) % 16)], 
		a[((nonce >> 16) % 16)], a[((nonce >> 12) % 16)], a[((nonce >> 8)  % 16)],
		a[((nonce >> 4)  % 16)], a[(nonce % 16)]
	};

	unsigned char *data = new unsigned char[payload.length() + 9];
    std::copy( payload.begin(), payload.end(), data );

	data[payload.length()+0] = nonce_byte[0];
	data[payload.length()+1] = nonce_byte[1];
	data[payload.length()+2] = nonce_byte[2];
	data[payload.length()+3] = nonce_byte[3];
	data[payload.length()+4] = nonce_byte[4];
	data[payload.length()+5] = nonce_byte[5];
	data[payload.length()+6] = nonce_byte[6];
	data[payload.length()+7] = nonce_byte[7];
	data[payload.length()+8] = 0;


    SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char *) data, payload.length()+8);	//ctx.state contains a-h

    sha256_final(&ctx, hash);
    
    
    return hash;
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
    return (hash[i] < difficulty[i]);
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