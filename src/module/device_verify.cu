#include "device_verify.cuh"
#include <iostream>
#include <sstream>
#include <boost/algorithm/hex.hpp>

using namespace std;

/* Module to verify transactions using CPU
 *
 * WARN: Verifying transaction using CPU takes a very long time
 */
bool CPU_verify(map<string, Transaction>& transaction_map) {
    for (map<string, Transaction>::iterator it = transaction_map.begin(); it != transaction_map.end(); it++) {
        Transaction& current_transaction = it->second;
        vector<Input> inputs = current_transaction.getInput();

        try {
            // Check every input against previous transactions outputs
            for (vector<Input>:: iterator vector_it = inputs.begin(); vector_it != inputs.end(); vector_it++) {
                if (vector_it->getTxID() != "0000000000000000000000000000000000000000000000000000000000000000") {
                    Transaction& previous_transaction = transaction_map[vector_it->getTxID()];
                
                    stringstream vout_stringstream;
                    unsigned long int vout;
                    vout_stringstream << hex << flip_hex_string_endian(vector_it->getVOUT());
                    vout_stringstream >> vout;

                    Output previous_transaction_output = previous_transaction.getOutput()[vout];
                    string script_public_key = previous_transaction_output.getScriptPubKey();
                    string script_signature = boost::algorithm::unhex(vector_it->getScriptSig());
                    string amount = previous_transaction_output.getValue();

                    CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey public_key = generate_public_key_from_string(script_public_key);
                    
                    if (!(verify_message(public_key, script_signature, amount))) {
                        string reason = "";
                        reason += "Signature mismatch\n";
                        reason += "    Signature    : " + script_signature + "\n";
                        reason += "    Public Key   : " + script_public_key + "\n";
                        reason += "    Payload      : " + amount + "\n";

                        throw reason;
                    }
                }
            }
        } catch (string e) {
            cout << e << endl;

            continue;
        }
    }

    return true;
}

/* Dispatch a verification based on user selection
 */
void device_verify_dispatcher(map<string, Transaction>& transaction_map, VerifyType reduction_type) {
    switch (reduction_type) {
        case VerifyType::CPU: {
            CPU_verify(transaction_map);
            break;
        }
    }
}