#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include "blockchain.hpp"
#include "utils.hpp"
#include "sha2.hpp"



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

template <typename T>
void
print_hash(char* buf, const T& hash)
{
    for (auto c : hash) {
        buf += sprintf(buf, "%02x", c);
    }
}

void Block::calculateHash(char *hash) {
    string payload_str = this->candidate_block.getHashableString();
    auto length = payload_str.length();
    const char *payload = payload_str.c_str();
    char data[length+8];
    for (uint32_t i = 0; i<length; i ++){
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
	sha2::sha256_hash result = sha2::sha256(ptr, length+8);
    print_hash(hash, result);
    
}

bool Block::verify_nonce(){
    char hash[129];
    this->calculateHash(hash);
    uint32_t i = 0;
    while (hash[i] == '0'){
        i++;
    }
    return i >= this->getDifficulty();
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