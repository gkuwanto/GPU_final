#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>

#define TRANSACTION_VERSION_SIZE    4
#define INPUT_TX_ID_SIZE            32
#define INPUT_VOUT_SIZE             4
#define INPUT_SEQUENCE_SIZE         4
#define OUTPUT_VALUE_SIZE           8
#define TRANSACTION_LOCKTIME_SIZE   4


struct Input {
    unsigned char tx_id[32];                // 32 bytes
    unsigned char v_out[4];                 // 4 bytes
    unsigned char script_sig_size_prefix;   // Determines the script_sig size
    unsigned char* script_sig_size;       // Up to 8 bytes
    unsigned char* script_sig;              // Variable, depends on script_sig_size
    unsigned char sequence[4];                  // 4 bytes
};

struct Output {
    unsigned char value[8];             // 8 bytes
    unsigned char script_pub_key_prefix;    // Determines the script_pub_key size
    unsigned char* script_pub_key_size;  // Up to 8 bytes
    unsigned char* script_pub_key;      // Variable, depends on script_pub_key_size
};

struct Transaction {
    unsigned char version[4];               // 4 bytes
    unsigned char input_count_prefix;       // Determines the input count
    unsigned char* input_count;           // Up to 8 bytes
    Input* input;                           // Transaction inputs
    unsigned char output_count_prefix;      // Determines the output count
    unsigned char* output_count;          // Up to 8 bytes
    Output* output;
    unsigned char locktime[4];
};

void parse_transaction(std::string raw_transaction, Transaction* transaction);
long long int insert_variable_integer(std::string raw_transaction, unsigned char* prefix, unsigned char** count, int* offset);

#endif