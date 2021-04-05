#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include "blockchain.hpp"
#include "utils.hpp"



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
    /* 
        Outputs:
        ###13 char for timestamp###
        ###64 char for previous hash###
        ### n char for transaction splitted with -###
    */
    stringstream ss;
    ss << this->timestamp << this->previous_block;
    for (vector<string>::iterator it = this->transaction_list.begin(); it != this->transaction_list.end(); it++) {
        ss << *it<<'-';
    }
    return ss.str();
}

Block::Block(CandidateBlock c_block, uint32_t nonce) {
    this->candidate_block = c_block;
    this->nonce = nonce;
}
string Block::calculateHash() {
    stringstream ss;
    ss << this->candidate_block.getHashableString() << hex << nonce;
    return hash_sha256(ss.str());
}

bool Block::verify_nonce(){
    string hash = this->calculateHash();
    uint32_t difficulty = this->candidate_block.getDifficulty();
    return hash.substr(0, difficulty) == string(difficulty, '0');
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
            throw "Block below difficulty level";
        }
        this->block_chain.push_back(new_block);
        
    } catch(string e) {
        cout << e << endl;
    }
}
