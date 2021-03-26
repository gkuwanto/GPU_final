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
unordered_map<string, Transaction> createTransactions(int n, vector<Address>& accounts);
string float_to_long_hex(float);
vector<Address> generateAccounts(int);
void integer_to_hex_string(int, string&, string&);

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

unordered_map<string, Transaction> createTransactions(int n, vector<Address>& accounts) {
    vector<Output> valid_outputs;
    unordered_map<string, Transaction> transactions;
    TransactionFactory factory;
    int number_of_accounts = accounts.size();

    // Set output randomizer
    random_device output_rng;
    mt19937 output_generator(output_rng());
    uniform_int_distribution<> output_distribution(1, 5);    

    // Set amount randomizer
    random_device amount_rng;
    mt19937 amount_generator(amount_rng());

    // Create the genesis transaction
    int number_of_outputs = output_distribution(output_rng);
    vector<Output> outputs;
    
    for (int i = 0; i < number_of_outputs; i++) {
        uniform_real_distribution<> amount_distribution(5.00, 30.00);
        
        Output output;
        output.setValue(float_to_long_hex(amount_distribution(amount_rng)));

        Address selected_address = accounts[output_distribution(output_rng)];
        stringstream public_key_stream;

        public_key_stream << hex << selected_address.getPublicKey().GetPublicElement().x;
        string x = public_key_stream.str();
        if (x.length() % 2 != 0) {
            x = "0" + x;
        }
        x.pop_back();

        public_key_stream.str(string());
        public_key_stream.clear();
        public_key_stream << hex << selected_address.getPublicKey().GetPublicElement().y;
        string y = public_key_stream.str();
        if (y.length() % 2 != 0) {
            y = "0" + y;
        }
        y.pop_back();

        string public_key = x + y;
        output.setScriptPubKey(public_key);

        string script_pub_key_prefix;
        string script_pub_key_size;
        integer_to_hex_string(public_key.length(), script_pub_key_prefix, script_pub_key_size);
        output.setScriptPubKeyPrefix(script_pub_key_prefix);
        output.setScriptPubKeySize(script_pub_key_size);
        
        outputs.push_back(output);
        valid_outputs.push_back(output);
    }

    Transaction transaction = factory.generateTransaction(outputs);

    pair<string, Transaction> pair(hash_sha256(hash_sha256(boost::algorithm::unhex(transaction.serialize()))), transaction);
    transactions.insert(pair);

    /* TODO: Child transactions and invalid transactions */

    return transactions;
}

string float_to_long_hex(float value) {
    stringstream stream;
    stream << setfill('0') << setw(sizeof(unsigned long long int)*2) << hex << (long long int) value * 100000000;
    
    return stream.str();
}

vector<Address> generateAccounts(int n) {
    vector<Address> accounts;

    for (int i = 0; i < n; i++) {
        accounts.push_back(Address());
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