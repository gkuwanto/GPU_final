#include <string>
#include <iostream>
#include <crypto++/osrng.h>
#include <crypto++/asn.h>
#include <crypto++/oids.h>
#include "classes.hpp"
#include "utils.hpp"

using namespace std;

Transaction::Transaction() { }
Transaction::Transaction(const Transaction& transaction) {
    this->version = transaction.version;
    this->input_count_prefix = transaction.version;
    this->input_count = transaction.input_count;
    
    for (int i = 0; i < transaction.input.size(); i++) {
        Input temp = transaction.input[i];
        this->input.push_back(temp);
    }

    this->output_count_prefix = transaction.output_count_prefix;
    this->output_count = transaction.output_count;
    for (int i = 0; i < transaction.output.size(); i++) {
        Output temp = transaction.output[i];
        this->output.push_back(temp);
    }

    this->locktime = transaction.locktime;
}
void Transaction::setTransaction(string raw_transaction) {
    int offset = 0;

    // Store the version
    this->version = raw_transaction.substr(offset, TRANSACTION_VERSION_SIZE);
    offset += TRANSACTION_VERSION_SIZE;
    
    // Store the input count
    insert_variable_integer(raw_transaction, this->input_count_prefix, this->input_count, &offset);

    // Iterate inputs
    long long int input_count = hex_string_to_long(this->input_count);
    for (long long int i = 0; i < input_count; i++) {
        Input input;
        
        input.setTxID(raw_transaction.substr(offset, INPUT_TX_ID_SIZE));
        offset += INPUT_TX_ID_SIZE;
        input.setVOUT(raw_transaction.substr(offset, INPUT_VOUT_SIZE));
        offset += INPUT_VOUT_SIZE;

        insert_variable_integer(raw_transaction, input.script_sig_size_prefix, input.script_sig_size, &offset);
        long long int script_sig_size = hex_string_to_long(input.script_sig_size);
        input.setScriptSig(raw_transaction.substr(offset, script_sig_size));
        offset += script_sig_size;

        input.setSequence(raw_transaction.substr(offset, INPUT_SEQUENCE_SIZE));
        offset += INPUT_SEQUENCE_SIZE;

        this->input.push_back(input);
    }

    // Store the output count
    insert_variable_integer(raw_transaction, this->output_count_prefix, this->output_count, &offset);

    // Iterate outputs
    long long int output_count = hex_string_to_long(this->output_count);
    
    for (long long int i = 0; i < output_count; i++) {
        Output output;

        output.setValue(raw_transaction.substr(offset, OUTPUT_VALUE_SIZE));
        offset += OUTPUT_VALUE_SIZE;

        insert_variable_integer(raw_transaction, output.script_pub_key_prefix, output.script_pub_key_size, &offset);
        long long int script_pub_key_size = hex_string_to_long(output.script_pub_key_size);
        output.setScriptPubKey(raw_transaction.substr(offset, script_pub_key_size));
        offset += script_pub_key_size;

        this->output.push_back(output);
    }

    // Store locktime
    this->locktime = raw_transaction.substr(offset, TRANSACTION_LOCKTIME_SIZE);
    offset += TRANSACTION_LOCKTIME_SIZE;

    cout << "Transaction size = " << offset << endl;
}
string Transaction::serialize() {
    string container = "";
    
    container += this->version;

    this->input_count_prefix.size() > 0 ? container += this->input_count_prefix : nullptr;
    container += this->input_count;
    for (vector<Input>::iterator it = this->input.begin(); it != input.end(); it++) {
        container += it->serialize();
    }

    this->output_count_prefix.size() > 0 ? container += this->output_count_prefix : nullptr;
    container += this->output_count;
    for (vector<Output>::iterator it = this->output.begin(); it != this->output.end(); it++) {
        container += it->serialize();
    }

    container += this->getLocktime();

    return container;
}
void Transaction::setVersion(string version) { this->version = version; }
void Transaction::setInputCountPrefix(string input_count_prefix) { this->input_count_prefix = input_count_prefix; }
void Transaction::setInputCount(string input_count) { this->input_count = input_count; }
void Transaction::setInput(const vector<Input>& input) { this->input = input; }
void Transaction::setOutputCountPrefix(string output_count_prefix) { this->output_count_prefix = output_count_prefix; }
void Transaction::setOutputCount(string output_count) { this->output_count = output_count; }
void Transaction::setOutput(const vector<Output>& output) { this->output = output; }
void Transaction::setLocktime(string locktime) { this->locktime = locktime; }
string Transaction::getVersion() { return this->version; }
string Transaction::getInputCountPrefix() { return this->input_count_prefix; }
string Transaction::getInputCount() { return this->input_count; }
vector<Input> Transaction::getInput() { return this->input; }
string Transaction::getOutputCountPrefix() { return this->output_count_prefix; }
string Transaction::getOutputCount() { return this->output_count; }
vector<Output> Transaction::getOutput() { return this->output; }
string Transaction::getLocktime() { return this->locktime; }


Input::Input() { }
Input::Input(const Input& input) {
    this->tx_id = input.tx_id;
    this->v_out = input.v_out;
    this->script_sig_size_prefix = input.script_sig_size_prefix;
    this->script_sig_size = input.script_sig_size;
    this->script_sig = input.script_sig;
    this->sequence = input.sequence;
}
string Input::serialize() {
    string container = "";

    container += this->tx_id;
    container += this->v_out;

    this->script_sig_size_prefix.size() > 0 ? container += this->script_sig_size_prefix : nullptr;
    container += this->script_sig_size;
    container += this->script_sig;
    container += this->sequence;

    return container;
}
void Input::setTxID(string tx_id) { this->tx_id = tx_id; }
void Input::setVOUT(string v_out) { this->v_out = v_out; }
void Input::setScriptSigSizePrefix(string script_sig_size_prefix) { this->script_sig_size_prefix = script_sig_size_prefix; }
void Input::setScriptSigSize(string script_sig_size) { this->script_sig_size = script_sig_size; }
void Input::setScriptSig(string script_sig) { this->script_sig = script_sig; }
void Input::setSequence(string sequence) { this->sequence = sequence; }
string Input::getTxID() { return this->tx_id; }
string Input::getVOUT() { return this->v_out; }
string Input::getScriptSigSizePrefix() { return this->script_sig_size_prefix; }
string Input::getScriptSigSize() { return this->script_sig_size; }
string Input::getScriptSig() { return this->script_sig; }
string Input::getSequence() { return this->sequence; }


Output::Output() { }
Output::Output(const Output& output) {
    this->value = output.value;
    this->script_pub_key_prefix = output.script_pub_key_prefix;
    this->script_pub_key_size = output.script_pub_key_size;
    this->script_pub_key = output.script_pub_key;
    this->sequence = output.sequence;
}
string Output::serialize() {
    string container = "";

    container += this->value;
    this->script_pub_key_prefix.size() > 0 ? container += this->script_pub_key_prefix : nullptr;
    container += this->script_pub_key_size;
    container += this->script_pub_key;

    return container;
}
void Output::setValue(string value) { this->value = value; }
void Output::setScriptPubKeyPrefix(string script_pub_key_prefix) { this->script_pub_key_prefix = script_pub_key_prefix; }
void Output::setScriptPubKeySize(string script_pub_key_size) { this->script_pub_key_size = script_pub_key_size; }
void Output::setScriptPubKey(string script_pub_key) { this->script_pub_key = script_pub_key; }
void Output::setSequence(string sequence) { this->sequence = sequence; }
string Output::getValue() { return this->value; }
string Output::getScriptPubKeyPrefix() { return this->script_pub_key_prefix; }
string Output::getScriptPubKeySize() { return this->script_pub_key_size; }
string Output::getScriptPubKey() { return this->script_pub_key; }
string Output::getSequence() { return this->sequence; }


Address::Address() {
    CryptoPP::AutoSeededRandomPool prng;
    
    this->private_key.Initialize(prng, CryptoPP::ASN1::secp160k1());
    this->private_key.MakePublicKey(this->public_key);
}