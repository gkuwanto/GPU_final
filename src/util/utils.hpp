#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>

#define TRANSACTION_VERSION_SIZE    4
#define INPUT_TX_ID_SIZE            32
#define INPUT_VOUT_SIZE             4
#define INPUT_SEQUENCE_SIZE         4
#define OUTPUT_VALUE_SIZE           8
#define TRANSACTION_LOCKTIME_SIZE   4

void insert_variable_integer(std::string raw_transaction, std::string& prefix, std::string& count, int* offset);
long long int hex_string_to_long(std::string hex_string);
std::string hash_sha256(const std::string& string);

#endif