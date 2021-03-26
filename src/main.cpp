#include "util/utils.hpp"
#include "util/classes.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <boost/algorithm/hex.hpp>

using namespace std;

int main(int argc, char** argv) {
    unordered_map<string, Transaction> transaction_map;

    ifstream finput("./transactions.txt");
    for (string line; getline(finput, line);) {
        Transaction temp_transaction;
        temp_transaction.setTransaction(boost::algorithm::unhex(line));
        
        pair<string, Transaction> temp_pair(hash_sha256(hash_sha256(boost::algorithm::unhex(line))), temp_transaction);
        transaction_map.insert(temp_pair);
    }
}