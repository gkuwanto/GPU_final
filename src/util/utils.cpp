#include "utils.hpp"
#include <random>
#include <sstream>
#include <iomanip>
#include <crypto++/sha.h>
#include <crypto++/hex.h>
#include <crypto++/cryptlib.h>
#include <boost/algorithm/hex.hpp>

using namespace std;


void insert_variable_integer(string raw_transaction, string& prefix, string& count, int* offset);
long long int hex_string_to_long(string hex_string);
string hash_sha256(const string& string);
string float_to_long_hex(float);
vector<Account> generate_accounts(int);
void integer_to_hex_string(int, string&, string&);
string flip_hex_string_endian(string);

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

string float_to_long_hex(float value) {
    stringstream stream;
    stream << setfill('0') << setw(sizeof(unsigned long long int)*2) << hex << (long long int) value * 100000000;
    
    return stream.str();
}

vector<Account> generate_accounts(int n) {
    vector<Account> accounts;

    for (int i = 0; i < n; i++) {
        accounts.push_back(Account());
    }

    return accounts;
}

void integer_to_hex_string(int value, string& prefix, string& size) {
    stringstream ss;
    ss << hex << value;
    string hex_string = ss.str();

    stringstream ss2;
    if (hex_string.length() <= 2) {
        ss2 << setfill('0') << setw(2) << hex << value;
        prefix = "";
    } else if (hex_string.length() <= 4) {
        ss2 << setfill('0') << setw(4) << hex << value;
        prefix = "FD";
    } else if (hex_string.length() <= 8) {
        ss2 << setfill('0') << setw(8) << hex << value;
        prefix = "FE";
    } else if (hex_string.length() <= 16) {
        ss2 << setfill('0') << setw(16) << hex << value;
        prefix = "FF";
    }
    size = ss2.str();
}

string flip_hex_string_endian(string value) {
    int bottom_pivot = 1;
    int upper_pivot = value.length() - 1;

    while (bottom_pivot < upper_pivot) {
        char temp;
        temp = value[bottom_pivot];
        value[bottom_pivot] = value[upper_pivot];
        value[upper_pivot] = temp;

        bottom_pivot--;
        upper_pivot--;

        temp = value[bottom_pivot];
        value[bottom_pivot] = value[upper_pivot];
        value[upper_pivot] = temp;

        bottom_pivot += 3;
        upper_pivot--;
    }

    return value;
}