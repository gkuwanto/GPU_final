#ifndef __CLASSES_HPP__
#define __CLASSES_HPP__

#include <string>
#include <tuple>
#include <vector>
#include <crypto++/eccrypto.h>

class Input {
    private:
        std::string tx_id;
        std::string v_out;
        std::string script_sig;
        std::string sequence;
    
    public:
        std::string script_sig_size_prefix;
        std::string script_sig_size;

        Input();
        Input(const Input&);
        std::string serialize();    /* Serializes input into hex string */
        std::string str();          /* Converts input into human-readable string */

        void setTxID(std::string);
        void setVOUT(std::string);
        void setScriptSigSizePrefix(std::string);
        void setScriptSigSize(std::string);
        void setScriptSig(std::string);
        void setSequence(std::string);
        std::string getTxID();
        std::string getVOUT();
        std::string getScriptSigSizePrefix();
        std::string getScriptSigSize();
        std::string getScriptSig();
        std::string getSequence();
};

class Output {
    private:
        std::string value;
        std::string script_pub_key;
        std::string sequence;
    
    public:
        std::string script_pub_key_prefix;
        std::string script_pub_key_size;
        
        Output();
        Output(const Output&);
        std::string serialize();    /* Serializes output into hex string */
        std::string str();          /* Converts output into human-readable string */

        void setValue(std::string);
        void setScriptPubKeyPrefix(std::string);
        void setScriptPubKeySize(std::string);
        void setScriptPubKey(std::string);
        void setSequence(std::string);
        std::string getValue();
        std::string getScriptPubKeyPrefix();
        std::string getScriptPubKeySize();
        std::string getScriptPubKey();
        std::string getSequence();
};

class Transaction {
    private:
        std::string version;
        std::vector<Input> input;
        std::vector<Output> output;
        std::string locktime;
    
    public:
        std::string input_count_prefix;
        std::string input_count;
        std::string output_count_prefix;
        std::string output_count;

        Transaction();
        Transaction(const Transaction&);
        std::string serialize();    /* Serializes transaction into hex string */
        std::string str();          /* Converts transaction, including all of the inputs and outputs into human-readable string */

        void setTransaction(std::string);
        void setVersion(std::string);
        void setInputCountPrefix(std::string);
        void setInputCount(std::string);
        void setInput(const std::vector<Input>&);
        void setOutputCountPrefix(std::string);
        void setOutputCount(std::string);
        void setOutput(const std::vector<Output>&);
        void setLocktime(std::string);
        std::string getVersion();
        std::string getInputCountPrefix();
        std::string getInputCount();
        std::vector<Input>& getInput();
        std::string getOutputCountPrefix();
        std::string getOutputCount();
        std::vector<Output>& getOutput();
        std::string getLocktime();
};

class Account {
    private:
        CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PrivateKey private_key;
        CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey public_key;
        std::vector<std::tuple<double, std::string, int>> coin_list;
    
    public:
        Account();

        Transaction payToAccount(std::vector<Account>&);    /* Pays into an account, generating transaction */
        std::string str();      /* Convert account data into human readable string without the private key */
        std::string str(bool);  /* Convert account data into human readable string. If bool set to true, prints private key as well */

        void setPrivateKey(CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PrivateKey&);
        void setPublicKey(CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey&);
        void setCoinList(std::vector<std::tuple<double, std::string, int>>&);
        CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PrivateKey& getPrivateKey();
        CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey& getPublicKey();
        std::string getPublicKeyStr();
        std::vector<std::tuple<double, std::string, int>>& getCoinList();
};

class Coinbase {
    public:
        Coinbase();

        Transaction payToAccount(std::vector<Account>&);    /* Generates COINBASE to accounts, returning transaction */
        std::string str();  /* Converts coinbase data into human readable string */
};

#endif