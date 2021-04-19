#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "classes.hpp"

/* Size is defined in bytes */
#define TRANSACTION_VERSION_SIZE    4
#define INPUT_TX_ID_SIZE            32
#define INPUT_VOUT_SIZE             4
#define INPUT_SEQUENCE_SIZE         4
#define OUTPUT_VALUE_SIZE           8
#define TRANSACTION_LOCKTIME_SIZE   4

#define NUMBER_OF_ACCOUNTS          10
#define NUMBER_OF_TRANSACTIONS      128    /* Number of transactions include both COINBASE and standard transactions */
#define COINBASE_DELAY              50      /* Number of transactions needed to generate a new COINBASE transaction */

void parse_variable_integer(std::string raw_transaction, std::string& prefix, std::string& count, int* offset);

std::string float_to_long_hex(float value);
long long int hex_string_to_long(std::string hex_string);
void variable_int_to_hex_string(int value, std::string& prefix, std::string& size);
std::string flip_hex_string_endian(std::string value);

std::vector<Account> generate_accounts(int n);

std::string hash_sha256(const std::string& string);
CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey generate_public_key_from_string(std::string public_key);
bool verify_message(CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey& public_key, std::string signature, std::string string);

#endif