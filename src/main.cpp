#include "util/utils.hpp"
#include "util/classes.hpp"
#include "util/blockchain.hpp"
#include "module/device_verify.cuh"
#include "module/device_mine.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <boost/algorithm/hex.hpp>

using namespace std;

cudaEvent_t start;
cudaEvent_t stop;
#define START_TIMER()        \
{                            \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);  \
    cudaEventRecord(start);  \
}
#define STOP_RECORD_TIMER(name)               \
{                                             \
    cudaEventRecord(stop);                    \
    cudaEventSynchronize(stop);               \
    cudaEventElapsedTime(&name, start, stop); \
    cudaEventDestroy(start);                  \
    cudaEventDestroy(stop);                   \
}

int main(int argc, char** argv) {
    vector<Account> accounts = generate_accounts(NUMBER_OF_ACCOUNTS);
    Coinbase coinbase;

    map<string, Transaction> transactions_map;
    int i = NUMBER_OF_TRANSACTIONS;

    Transaction transaction = coinbase.payToAccount(accounts);
    pair<string, Transaction> pair(flip_hex_string_endian(hash_sha256(hash_sha256(boost::algorithm::unhex(transaction.serialize())))), transaction);
    transactions_map.insert(pair);

    i--;
    int round_robin_index = 0;
    while (i > 0) {
        if (accounts[round_robin_index].getCoinList().size() > 0) {
            transaction = accounts[round_robin_index].payToAccount(accounts);
            
            pair = std::pair<string, Transaction>(flip_hex_string_endian(hash_sha256(hash_sha256(boost::algorithm::unhex(transaction.serialize())))), transaction);

            transactions_map.insert(pair);
            i--;
        }

        round_robin_index = (round_robin_index + 1) % NUMBER_OF_ACCOUNTS;
        if (i % COINBASE_DELAY == 0) {
            transaction = coinbase.payToAccount(accounts);
            pair = std::pair<string, Transaction>(flip_hex_string_endian(hash_sha256(hash_sha256(boost::algorithm::unhex(transaction.serialize())))), transaction);

            transactions_map.insert(pair);
            i--;
        }
    }

    // cout << "Account List" << endl << "------------------------------" << endl;
    // for (int i = 0; i < accounts.size(); i++) {
    //     cout << accounts[i].str(true) << endl;;
    // }
    // cout << endl;
    
    VerifyType v_type = VerifyType::CPU;
    float time;
    START_TIMER();
    device_verify_dispatcher(transactions_map, v_type);
    STOP_RECORD_TIMER(time);

    cout << "Time spent to verify using CPU: " << time << "ms" << endl;

    vector<string> tx_list;
    for (map<string, Transaction>::iterator it = transactions_map.begin(); it != transactions_map.end(); it++) {
        Transaction& current_transaction = it->second;
        tx_list.push_back(current_transaction.serialize());
    }
    Blockchain blockchain;
    uint32_t diff = blockchain.getDifficulty();
    CandidateBlock candidate_block(diff);
    candidate_block.setTransactionList(tx_list);
    candidate_block.setPreviousBlock(hash_sha256("0")); // Genesis Block is "0"
    string payload = candidate_block.getHashableString();

    MineType m_type_1 = MineType::MINE_CPU;
    float mine_time_cpu;
    START_TIMER();
    int nonce = device_mine_dispatcher(payload, diff, m_type_1);
    STOP_RECORD_TIMER(mine_time_cpu);
    cout << "Time spent to mine using CPU: " << mine_time_cpu << "ms" << " with nonce:" << nonce << endl;

    
    MineType m_type_2 = MineType::MINE_GPU;
    float mine_time_gpu;
    START_TIMER();
    int nonce_gpu = device_mine_dispatcher(payload, diff, m_type_2);
    STOP_RECORD_TIMER(mine_time_gpu);
    cout << "Time spent to mine using GPU: " << mine_time_gpu << "ms" << " with nonce:" << nonce_gpu << endl;

    Block block(candidate_block, nonce);
    Block block_gpu(candidate_block, nonce_gpu);
    cout << block_gpu.verify_nonce();
    blockchain.addBlock(block);
    ofstream ofs("output.txt");
    ofs << blockchain.str();


    return 0;
}