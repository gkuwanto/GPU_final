#include "util/utils.hpp"
#include "util/classes.hpp"
#include <string>
#include <unordered_map>

using namespace std;

int main(int argc, char** argv) {
    vector<Address> accounts = generateAccounts(ACCOUNT_NUMBER);

    unordered_map<string, Transaction> transaction_map = createTransactions(10, accounts);

    return 0;
}