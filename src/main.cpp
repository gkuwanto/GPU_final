#include "util/utils.hpp"
#include "util/classes.hpp"
#include "module/device_verify.hpp"
#include <iostream>
#include <string>
#include <map>
#include <boost/algorithm/hex.hpp>

using namespace std;

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
    
    VerifyType type = VerifyType::CPU;
    device_verify_dispatcher(transactions_map, type);
    
    return 0;
}