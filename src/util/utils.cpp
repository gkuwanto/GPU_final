#include "utils.hpp"
#include <sstream>
#include <crypto++/sha.h>
#include <crypto++/hex.h>
#include <crypto++/cryptlib.h>

using namespace std;


void insert_variable_integer(string raw_transaction, string& prefix, string& count, int* offset);
long long int hex_string_to_long(string hex_string);
string hash_sha256(const string& string);

void insert_variable_integer(string raw_transaction, string& prefix, string& count, int* offset) {
    prefix = raw_transaction.substr(*offset, 1);
    (*offset)++;

    if (prefix == "FD") {
        count = raw_transaction.substr(*offset, 2);
        prefix = "FD";
        (*offset) += 2;
    } else if (prefix == "FE") {
        count = raw_transaction.substr(*offset, 4);
        prefix = "FE";
        (*offset) += 4;
    } else if (prefix == "FF") {
        count = raw_transaction.substr(*offset, 8);
        prefix = "FF";
        (*offset) += 8;
    } else {
        count = prefix;
        prefix = "";
    }
}

long long int hex_string_to_long(string hex_string) {
    const char* temp_hex_string = hex_string.c_str();
    stringstream ss;
    ss << hex;

    for (int i = 0; i < hex_string.length(); i++) {
        ss << (unsigned int) temp_hex_string[i];
    }

    long long int result;
    ss >> result;
    return result;
}

string hash_sha256(const string& payload) {
    byte digest[CryptoPP::SHA256::DIGESTSIZE];
    CryptoPP::SHA256().CalculateDigest(digest, (byte *) payload.c_str(), payload.size());

    string hashed_payload;
    CryptoPP::HexEncoder encoder;
    encoder.Attach(new CryptoPP::StringSink(hashed_payload));
    encoder.Put(digest, sizeof(digest));
    encoder.MessageEnd();

    return hashed_payload;
}