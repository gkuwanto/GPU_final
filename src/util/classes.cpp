#include <random>
#include <sstream>
#include <boost/algorithm/hex.hpp>
#include <crypto++/asn.h>
#include <crypto++/oids.h>
#include <crypto++/osrng.h>
#include "classes.hpp"
#include "utils.hpp"

using namespace std;

Transaction::Transaction() { }
Transaction::Transaction(const Transaction& transaction) {
    this->version = transaction.version;
    this->input_count_prefix = transaction.version;
    this->input_count = transaction.input_count;
    
    for (unsigned int i = 0; i < transaction.input.size(); i++) {
        Input temp = transaction.input[i];
        this->input.push_back(temp);
    }

    this->output_count_prefix = transaction.output_count_prefix;
    this->output_count = transaction.output_count;
    for (unsigned int i = 0; i < transaction.output.size(); i++) {
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
    parse_variable_integer(raw_transaction, this->input_count_prefix, this->input_count, &offset);

    // Iterate inputs
    long long int input_count = hex_string_to_long(this->input_count);
    for (long long int i = 0; i < input_count; i++) {
        Input input;
        
        input.setTxID(raw_transaction.substr(offset, INPUT_TX_ID_SIZE));
        offset += INPUT_TX_ID_SIZE;
        input.setVOUT(raw_transaction.substr(offset, INPUT_VOUT_SIZE));
        offset += INPUT_VOUT_SIZE;

        parse_variable_integer(raw_transaction, input.script_sig_size_prefix, input.script_sig_size, &offset);
        long long int script_sig_size = hex_string_to_long(input.script_sig_size);
        input.setScriptSig(raw_transaction.substr(offset, script_sig_size));
        offset += script_sig_size;

        input.setSequence(raw_transaction.substr(offset, INPUT_SEQUENCE_SIZE));
        offset += INPUT_SEQUENCE_SIZE;

        this->input.push_back(input);
    }

    // Store the output count
    parse_variable_integer(raw_transaction, this->output_count_prefix, this->output_count, &offset);

    // Iterate outputs
    long long int output_count = hex_string_to_long(this->output_count);
    
    for (long long int i = 0; i < output_count; i++) {
        Output output;

        output.setValue(raw_transaction.substr(offset, OUTPUT_VALUE_SIZE));
        offset += OUTPUT_VALUE_SIZE;

        parse_variable_integer(raw_transaction, output.script_pub_key_prefix, output.script_pub_key_size, &offset);
        long long int script_pub_key_size = hex_string_to_long(output.script_pub_key_size);
        output.setScriptPubKey(raw_transaction.substr(offset, script_pub_key_size));
        offset += script_pub_key_size;

        this->output.push_back(output);
    }

    // Store locktime
    this->locktime = raw_transaction.substr(offset, TRANSACTION_LOCKTIME_SIZE);
    offset += TRANSACTION_LOCKTIME_SIZE;
}
string Transaction::serialize() {
    string container = "";
    container += this->version;

    if (this->input_count_prefix.length() > 0) {
        container += this->input_count_prefix;
    }
    container += this->input_count;
    for (vector<Input>::iterator it = this->input.begin(); it != input.end(); it++) {
        container += it->serialize();
    }

    if (this->input_count_prefix.length() > 0) {
        container += this->output_count_prefix;
    }
    container += this->output_count;
    for (vector<Output>::iterator it = this->output.begin(); it != this->output.end(); it++) {
        container += it->serialize();
    }

    container += this->getLocktime();

    return container;
}
string Transaction::str() {
    string container;

    container += "Version       " + version + "\n";
    container += "Input Count   " + input_count + "\n";
    container += "Inputs\n";
    for (unsigned int i = 0; i < input.size(); i++) {
        container += "      " + to_string(i + 1) + ".  Transaction ID  " + input[i].getTxID() + "\n";
        container += "          V_OUT           " + input[i].getVOUT() + "\n";
        container += "          ScriptSig Size  " + input[i].getScriptSigSize() + "\n";
        container += "          ScriptSig       " + input[i].getScriptSig() + "\n";
        container += "          Sequence        " + input[i].getSequence() + "\n";
    }
    container += "Output Count  " + output_count + "\n";
    container += "Outputs\n";
    for (unsigned int i = 0; i < output.size(); i++) {
        container += "       " + to_string(i + 1) + ".  Value               " + output[i].getValue() + "\n";
        container += "          ScriptPubKey Size   " + output[i].getScriptPubKeySize() + "\n";
        container += "          ScriptPubKey        " + output[i].getScriptPubKey() + "\n";
        container += "          Sequence            " + output[i].getSequence() + "\n";
    }
    container += "Locktime      " + locktime + "\n";

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
vector<Input>& Transaction::getInput() { return this->input; }
string Transaction::getOutputCountPrefix() { return this->output_count_prefix; }
string Transaction::getOutputCount() { return this->output_count; }
vector<Output>& Transaction::getOutput() { return this->output; }
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

    if (this->script_sig_size_prefix.length() > 0) {
        container += this->script_sig_size_prefix;
    }
    container += this->script_sig_size;
    container += this->script_sig;
    container += this->sequence;

    return container;
}
string Input::str() {
    string container;

    container += "Transaction ID  " + tx_id + "\n";
    container += "V_OUT           " + v_out + "\n";
    container += "ScriptSig Size  " + script_sig_size + "\n";
    container += "ScriptSig       " + script_sig + "\n";
    container += "Sequence        " + sequence + "\n";

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
    
    if (this-script_pub_key_prefix.length() > 0) {
        this->script_pub_key_prefix = output.script_pub_key_prefix;
    }
    this->script_pub_key_size = output.script_pub_key_size;
    this->script_pub_key = output.script_pub_key;
    this->sequence = output.sequence;
}
string Output::serialize() {
    string container = "";

    container += this->value;
    container += this->script_pub_key_prefix;
    container += this->script_pub_key_size;
    container += this->script_pub_key;

    return container;
}
string Output::str() {
    string container;

    container += "Value               " + value + "\n";
    container += "ScriptPubKey Size   " + script_pub_key_size + "\n";
    container += "ScriptPubKey        " + script_pub_key + "\n";
    container += "Sequence            " + sequence + "\n";

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


Account::Account() {
    CryptoPP::AutoSeededRandomPool prng;
    
    this->private_key.Initialize(prng, CryptoPP::ASN1::secp256k1());
    this->private_key.MakePublicKey(this->public_key);
}
void Account::setPrivateKey(CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PrivateKey& private_key) {
    this->private_key = private_key;
}
void Account::setPublicKey(CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey& public_key) {
    this->public_key = public_key;
}
void Account::setCoinList(vector<std::tuple<double, string, int>>& coin_list) {
    this->coin_list = coin_list;
}
CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey& Account::getPublicKey() {
    return this->public_key;
}
CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PrivateKey& Account::getPrivateKey() {
    return this->private_key;
}
vector<std::tuple<double, string, int>>& Account::getCoinList() {
    return this->coin_list;
}
string Account::str() {
    string container;

    stringstream public_key_stream;
    
    public_key_stream << hex << public_key.GetPublicElement().x;
    string x = public_key_stream.str();
    x.pop_back();
    if (x.length() % 2 != 0) {
        x = "0" + x;
    }
    public_key_stream.str(string());
    public_key_stream.clear();
    public_key_stream << hex << public_key.GetPublicElement().y;
    string y = public_key_stream.str();
    y.pop_back();
    if (y.length() % 2 != 0) {
        y = "0" + y;
    }
    container += "Public Key    " + x + y + "\n";
    container += "Coins\n";
    for (unsigned int i = 0; i < coin_list.size(); i++) {
        tuple<double, string, int> tuple = coin_list[i];
        container += "      " + to_string(i + 1) + ".  Amount          " + to_string(get<0>(tuple)) + "\n";
        container += "          Transaction ID  " + get<1>(tuple) + "\n";
        container += "          V_OUT           " + to_string(get<2>(tuple)) + "\n";
    }

    return container;
}
string Account::str(bool show_private_key) {
    stringstream ss;

    ss << "Private Key   " << hex << private_key.GetPrivateExponent() << endl;
    
    return ss.str() + str();
}
Transaction Account::payToAccount(vector<Account>& accounts) {
    // Initialize transaction
    Transaction transaction;
    transaction.setVersion("01000000");
    
    // Set input count randomizer
    random_device input_count_rng;
    mt19937 input_count_generator(input_count_rng());
    uniform_int_distribution<> input_count_distribution(1, coin_list.size());

    // Set input select randomizer
    random_device input_select_rng;
    mt19937 input_select_generator(input_select_rng());

    // Set output count randomizer
    random_device output_count_rng;
    mt19937 output_count_generator(output_count_rng());
    uniform_int_distribution<> output_count_distribution(1, accounts.size());

    // Set output select randomizer
    random_device output_select_rng;
    mt19937 output_select_generator(output_select_rng());
    uniform_int_distribution<> output_select_distribution(0, accounts.size() - 1);

    // Set amount randomizer
    random_device amount_rng;
    mt19937 amount_generator(amount_rng());

    // Create inputs
    double spendable_coins = 0;
    int number_of_inputs = input_count_distribution(input_count_rng);
    vector<Input> inputs;
    for (int i = 0; i < number_of_inputs; i++) {
        uniform_int_distribution<> input_select_distribution(0, coin_list.size() - 1);

        int selected_input_index = input_select_distribution(input_select_rng);

        tuple<double, string, int> selected_input = coin_list[selected_input_index];
        coin_list.erase(coin_list.begin() + selected_input_index);

        Input input;
        input.setTxID(flip_hex_string_endian(get<1>(selected_input)));
        
        stringstream ss;
        ss << setfill('0') << setw(8) << hex << get<2>(selected_input);
        input.setVOUT(flip_hex_string_endian(ss.str()));

        string signature;
        CryptoPP::AutoSeededRandomPool prng;
        CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::Signer signer(private_key);
        CryptoPP::StringSource s(flip_hex_string_endian(float_to_long_hex(get<0>(selected_input))), true, new CryptoPP::SignerFilter(prng, signer, new CryptoPP::StringSink(signature)));

        input.setScriptSig(boost::algorithm::hex(signature));
        string script_sig_size_prefix;
        string script_sig_size;
        variable_int_to_hex_string(signature.length(), script_sig_size_prefix, script_sig_size);
        input.setScriptSigSizePrefix(script_sig_size_prefix);
        input.setScriptSigSize(script_sig_size);
        input.setSequence("FFFFFFFF");

        spendable_coins += get<0>(selected_input);
        inputs.push_back(input);
    }
    string input_count_prefix;
    string input_count;
    variable_int_to_hex_string(inputs.size(), input_count_prefix, input_count);
    transaction.setInput(inputs);
    transaction.setInputCountPrefix(input_count_prefix);
    transaction.setInputCount(input_count);

    // Create outputs
    int number_of_outputs = output_count_distribution(output_count_rng);
    vector<Output> outputs;
    vector<tuple<Account*, double, int>> unverified_coins_list;
    for (int i = 0; i < number_of_outputs; i++) {
        int selected_account_index = output_select_distribution(output_select_rng);
        
        Account& selected_account = accounts[selected_account_index];

        Output output;

        double amount;
        if (i - 1 != number_of_outputs) {
            uniform_real_distribution<> amount_distribution(0.00, spendable_coins);

            amount = amount_distribution(amount_rng);
            spendable_coins -= amount;
        } else {
            amount = spendable_coins;
        }
        output.setValue(flip_hex_string_endian(float_to_long_hex(amount)));

        stringstream public_key_stream;
        public_key_stream << hex << selected_account.getPublicKey().GetPublicElement().x;
        string x = public_key_stream.str();
        x.pop_back();
        if (x.length() % 2 != 0) {
            x = "0" + x;
        }

        public_key_stream.str(string());
        public_key_stream.clear();
        public_key_stream << hex << selected_account.getPublicKey().GetPublicElement().y;
        string y = public_key_stream.str();
        y.pop_back();
        if (y.length() % 2 != 0) {
            y = "0" + y;
        }

        string public_key = x + y;
        output.setScriptPubKey(public_key);

        string script_pub_key_prefix;
        string script_pub_key_size;
        variable_int_to_hex_string(public_key.length(), script_pub_key_prefix, script_pub_key_size);
        output.setScriptPubKeyPrefix(script_pub_key_prefix);
        output.setScriptPubKeySize(script_pub_key_size);

        outputs.push_back(output);
        unverified_coins_list.push_back(tuple<Account*, double, int>(&selected_account, amount, i));
    }

    string output_count_prefix;
    string output_count;
    variable_int_to_hex_string(outputs.size(), output_count_prefix, output_count);
    transaction.setOutputCountPrefix(output_count_prefix);
    transaction.setOutputCount(output_count);
    transaction.setOutput(outputs);

    transaction.setLocktime("00000000");

    // Set TX ID to outputs
    string tx_id = hash_sha256(hash_sha256(boost::algorithm::unhex(transaction.serialize())));
    for (unsigned int i = 0; i < unverified_coins_list.size(); i++) {
        tuple<Account*, double, int> tuple = unverified_coins_list[i];
        
        Account* account = get<0>(tuple);
        double amount = get<1>(tuple);
        int vout = get<2>(tuple);

        std::tuple<double, string, int> verified_coin(amount, tx_id, vout);
        (*account).getCoinList().push_back(verified_coin);
    }

    return transaction;
}

Coinbase::Coinbase() { }
string Coinbase::str() {
    string container;

    container += "COINBASE Account\n";
    container += "Special ScriptSig inside!\n";
    return container;
}
Transaction Coinbase::payToAccount(vector<Account>& accounts) {
    // Initialize transaction
    Transaction transaction;
    transaction.setVersion("01000000");
    transaction.setInputCountPrefix("");
    transaction.setInputCount("01");

    // Set output count randomizer
    random_device output_count_rng;
    mt19937 output_count_generator(output_count_rng());
    uniform_int_distribution<> output_count_distribution(1, accounts.size());

    // Set output selector randomizer
    random_device output_select_rng;
    mt19937 output_select_generator(output_select_rng());
    uniform_int_distribution<> output_select_distribution(0, accounts.size() - 1);

    // Set amount randomizer
    random_device amount_rng;
    mt19937 amount_generator(amount_rng());
    uniform_real_distribution<> amount_distribution(5.00, 30.00);

    // Create input
    Input input;
    input.setTxID("0000000000000000000000000000000000000000000000000000000000000000");
    input.setVOUT("FFFFFFFF");
    input.setScriptSigSizePrefix("");
    input.setScriptSigSize("18");
    input.setScriptSig("494634303434202D204750552050524F4752414D4D494E47");
    input.setSequence("FFFFFFFF");
    transaction.getInput().push_back(input);

    // Create outputs
    int number_of_outputs = output_count_distribution(output_count_rng);
    vector<Output> outputs;
    vector<tuple<Account*, double, int>> unverified_coins_list;
    for (int i = 0; i < number_of_outputs; i++) {
        Account& selected_account = accounts[output_select_distribution(output_select_rng)];

        Output output;
        double amount = amount_distribution(amount_rng);
        output.setValue(flip_hex_string_endian(float_to_long_hex(amount)));

        stringstream public_key_stream;

        public_key_stream << hex << selected_account.getPublicKey().GetPublicElement().x;
        string x = public_key_stream.str();
        x.pop_back();
        if (x.length() % 2 != 0) {
            x = "0" + x;
        }

        public_key_stream.str(string());
        public_key_stream.clear();
        public_key_stream << hex << selected_account.getPublicKey().GetPublicElement().y;
        string y = public_key_stream.str();
        y.pop_back();
        if (y.length() % 2 != 0) {
            y = "0" + y;
        }

        string public_key = x + y;
        output.setScriptPubKey(public_key);

        string script_pub_key_prefix;
        string script_pub_key_size;
        variable_int_to_hex_string(public_key.length(), script_pub_key_prefix, script_pub_key_size);
        output.setScriptPubKeyPrefix(script_pub_key_prefix);
        output.setScriptPubKeySize(script_pub_key_size);

        outputs.push_back(output);
        unverified_coins_list.push_back(tuple<Account*, double, int>(&selected_account, amount, i));
    }

    string output_count_prefix;
    string output_count;
    variable_int_to_hex_string(outputs.size(), output_count_prefix, output_count);
    transaction.setOutputCountPrefix(output_count_prefix);
    transaction.setOutputCount(output_count);
    transaction.setOutput(outputs);

    transaction.setLocktime("00000000");

    // Set TX ID to outputs
    string tx_id = hash_sha256(hash_sha256(boost::algorithm::unhex(transaction.serialize())));
    for (unsigned int i = 0; i < unverified_coins_list.size(); i++) {
        tuple<Account*, double, int> tuple = unverified_coins_list[i];
        
        Account* account = get<0>(tuple);
        double amount = get<1>(tuple);
        int vout = get<2>(tuple);

        std::tuple<double, string, int> verified_coin(amount, tx_id, vout);
        (*account).getCoinList().push_back(verified_coin);
    }

    return transaction;
}