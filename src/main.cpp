#include "util/utils.hpp"
#include "util/classes.hpp"
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main(int argc, char** argv) {
    vector<Account> accounts = generate_accounts(NUMBER_OF_ACCOUNTS);
    Coinbase coinbase;

    vector<Transaction> transactions;
    int i = NUMBER_OF_TRANSACTIONS;

    transactions.push_back(coinbase.payToAccount(accounts));
    i--;
    int round_robin_index = 0;
    while (i > 0) {
        if (accounts[round_robin_index].getCoinList().size() > 0) {
            transactions.push_back(accounts[round_robin_index].payToAccount(accounts));
            i--;
        }

        round_robin_index = (round_robin_index + 1) % NUMBER_OF_ACCOUNTS;
        if (i % COINBASE_DELAY == 0) {
            transactions.push_back(coinbase.payToAccount(accounts));
        }
    }

    cout << "Account List" << endl << "------------------------------" << endl;
    for (int i = 0; i < accounts.size(); i++) {
        cout << accounts[i].str(true) << endl;;
    }
    cout << endl;

    return 0;
}