#include "utils.hpp"
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>

using namespace std;

long long int insert_variable_integer(string raw_transaction, unsigned char* prefix, unsigned char** count, int* offset);
void parse_transaction(string raw_transaction, Transaction* transaction);

long long int insert_variable_integer(string raw_transaction, unsigned char* prefix, unsigned char** count, int* offset) {
    unsigned char variable_count = 0x00;
    raw_transaction.copy((char *) &variable_count, sizeof(char), (*offset));
    (*offset)++;

    stringstream ss;
    ss << hex;
    switch (variable_count) {
        case 0xFD: {
            *prefix = 0xFD;
            *count = (unsigned char *) malloc (2 * sizeof(char));
            raw_transaction.copy((char *) *count, 2 * sizeof(char), (*offset));
            
            for (int i = 0; i < 2; i++) {
                ss << (unsigned int) raw_transaction[*offset];
                (*offset)++;
            }
            break;

        }
        case 0xFE: {
            *prefix = 0xFE;
            *count = (unsigned char *) malloc (4 * sizeof(char));
            raw_transaction.copy((char *) *count, 4 * sizeof(char), (*offset));

            for (int i = 0; i < 4; i++) {
                ss << (unsigned int) raw_transaction[*offset];
                (*offset)++;
            }
            break;
        }
        case 0xFF: {
            *prefix = 0xFF;
            *count = (unsigned char *) malloc (8 * sizeof(char));
            raw_transaction.copy((char *) *count, 8 * sizeof(char), (*offset));

            for (int i = 0; i < 8; i++) {
                ss << (unsigned int) raw_transaction[*offset];
                (*offset)++;
            }
            break;
        }
        default: {
            *count = (unsigned char *) malloc (sizeof(char));
            *prefix = 0x00;
            **count = variable_count;

            for (int i = 0; i < 1; i++) {
                ss << (unsigned int) raw_transaction[*offset - 1];
            }
            break;
        }
    }

    long long int temp;
    ss >> temp;

    return temp;
}
void parse_transaction(string raw_transaction, Transaction* transaction) {
    int offset = 0; // Stores the byte offset from raw_transaction

    // Store the version
    raw_transaction.copy((char *) transaction->version, TRANSACTION_VERSION_SIZE, offset);
    offset += TRANSACTION_VERSION_SIZE;

    // Store the input count
    long long int input_count_long = insert_variable_integer(raw_transaction, &transaction->input_count_prefix, &transaction->input_count, &offset);

    // Iterate inputs
    transaction->input = (Input *) malloc (input_count_long * sizeof(Input));
    for (long long int i = 0; i < input_count_long; i++) {
        raw_transaction.copy((char *) transaction->input[i].tx_id, INPUT_TX_ID_SIZE, offset);
        offset += INPUT_TX_ID_SIZE;
        raw_transaction.copy((char *) transaction->input[i].v_out, INPUT_VOUT_SIZE, offset);
        offset += INPUT_VOUT_SIZE;
        
        long long int script_sig_size_long = insert_variable_integer(raw_transaction, &transaction->input[i].script_sig_size_prefix, &transaction->input[i].script_sig_size, &offset);
        transaction->input[i].script_sig = (unsigned char *) malloc (script_sig_size_long * sizeof(char));
        raw_transaction.copy((char *) transaction->input[i].script_sig, script_sig_size_long, offset);
        offset += script_sig_size_long;

        raw_transaction.copy((char *) transaction->input[i].sequence, INPUT_SEQUENCE_SIZE, offset);
        offset += INPUT_SEQUENCE_SIZE;
    }

    // Store the output count
    long long int output_count_long = insert_variable_integer(raw_transaction, &transaction->output_count_prefix, &transaction->output_count, &offset);

    // Iterate outputs
    transaction->output = (Output *) malloc (output_count_long * sizeof(Output));
    for (long long int i = 0; i < output_count_long; i++) {
        raw_transaction.copy((char *) transaction->output[i].value, OUTPUT_VALUE_SIZE, offset);
        offset += OUTPUT_VALUE_SIZE;
        
        long long int script_pub_key_size_long = insert_variable_integer(raw_transaction, &transaction->output[i].script_pub_key_prefix, &transaction->output[i].script_pub_key_size, &offset);
        transaction->output[i].script_pub_key = (unsigned char *) malloc (script_pub_key_size_long * sizeof(char));
        raw_transaction.copy((char *) transaction->output[i].script_pub_key, script_pub_key_size_long, offset);
        offset += script_pub_key_size_long;
    }

    // Store locktime
    raw_transaction.copy((char *) transaction->locktime, TRANSACTION_LOCKTIME_SIZE, offset);
    offset += TRANSACTION_LOCKTIME_SIZE;

    cout << "Transaction size = " << offset << endl;
}