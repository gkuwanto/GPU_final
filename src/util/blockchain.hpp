#ifndef __BLOCKCHAIN_HPP__
#define __BLOCKCHAIN_HPP__

#include <string>
#include <vector>
/*
    A simple implementation of a Blockchain without a merkle root.
*/
class CandidateBlock {
    private:
        uint64_t timestamp
        std::string previous_block; /* Hash of previous block */
        std::vector<std::string> transaction_list; /* List of transactions hash to be added */
        uint32_t difficulty;
    public:
        CandidateBlock();
        CandidateBlock(uint32_t difficulty);
        void setPreviousBlock(std::string prev_hash);
        void setTransactionList(std::vector<std::string>& tx_list);
        uint32_t getDifficulty();
        std::string getHashableString();
};

class Block {
    private:
        CandidateBlock candidate_block;
        uint32_t nonce;
    public:
        std::string calculateHash();
        Block(CandidateBlock, uint32_t);
        bool verify_nonce();   
};

class Blockchain {
    private:
        uint32_t current_difficulty;
        std::vector<Block> block_chain;
    public:
        Blockchain();
        uint32_t getDifficulty();
        void addBlock(Block new_block);
};

#endif