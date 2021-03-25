#include "utils.hpp"
#include <iostream>
#include <boost/algorithm/hex.hpp>

using namespace std;

int main(int argc, char** argv) {
    string transaction_hex = boost::algorithm::unhex(string("01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff4503ec59062f48616f4254432f53756e204368756e2059753a205a6875616e67205975616e2c2077696c6c20796f75206d61727279206d653f2f06fcc9cacc19c5f278560300ffffffff01529c6d98000000001976a914bfd3ebb5485b49a6cf1657824623ead693b5a45888ac00000000"));

    Transaction transaction;
    parse_transaction(transaction_hex, &transaction);
}