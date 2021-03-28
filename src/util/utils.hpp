#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include <unordered_map>
#include <vector>
#include "classes.hpp"

#define TRANSACTION_VERSION_SIZE    4
#define INPUT_TX_ID_SIZE            32
#define INPUT_VOUT_SIZE             4
#define INPUT_SEQUENCE_SIZE         4
#define OUTPUT_VALUE_SIZE           8
#define TRANSACTION_LOCKTIME_SIZE   4

#define NUMBER_OF_ACCOUNTS 10
#define NUMBER_OF_TRANSACTIONS 1000
#define COINBASE_DELAY 50

void insert_variable_integer(std::string raw_transaction, std::string& prefix, std::string& count, int* offset);
long long int hex_string_to_long(std::string hex_string);
std::string hash_sha256(const std::string& string);
std::string float_to_long_hex(float);
std::vector<Account> generate_accounts(int);
void integer_to_hex_string(int, std::string&, std::string&);
std::string flip_hex_string_endian(std::string);
CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey generate_public_key_from_string(std::string);
bool verify_signature(CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey&, std::string, std::string);

#endif