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
            for (int i = 0; i < inputs.size(); i++) {
                if (inputs[i].getTxID() != "0000000000000000000000000000000000000000000000000000000000000000") {
                    Transaction& previous_transaction = transaction_map[inputs[i].getTxID()];
                
                    stringstream vout_stringstream;
                    unsigned long int vout;
                    vout_stringstream << hex << flip_hex_string_endian(inputs[i].getVOUT());
                    vout_stringstream >> vout;
                    Output previous_transaction_output = previous_transaction.getOutput()[vout];
                    
                    // Set up payload for verifying output
                    Transaction temp_transaction(current_transaction);
                    vector<Input> temp_inputs;
                    string current_signature;
                    string hash_type_code;
        
                    for (int j = 0; j < inputs.size(); j++) {
                        if (i == j) {
                            Input input(inputs[i]);

                            current_signature = input.getScriptSig();
                            hash_type_code = current_signature.substr(current_signature.length()-2, 2);
                            current_signature.erase(current_signature.length()-2, 2);
                            input.setScriptSig(previous_transaction_output.getScriptPubKey());
                            input.setScriptSigSize(previous_transaction_output.getScriptPubKeySize());
                            input.setScriptSigSizePrefix(previous_transaction_output.getScriptPubKeyPrefix());
                            input.setSequence("FFFFFFFF");
                            
                            temp_inputs.push_back(input);
                        } else {
                            Input input;
                            input.setTxID(inputs[i].getTxID());
                            input.setVOUT(inputs[i].getVOUT());
                            input.setScriptSig("");
                            input.setScriptSigSize("00");
                            input.setScriptSigSizePrefix("");
                            input.setSequence("FFFFFFFF");

                            temp_inputs.push_back(input);
                        }
                    }

                    // Set input to temp transaction, and serialize
                    temp_transaction.setInput(temp_inputs);
                    string serialized_temp_transaction = temp_transaction.serialize();

                    // Append SIGHASH 0x01
                    serialized_temp_transaction += "01000000";

                    // Hash twice with SHA256
                    string hashed_serialized_temp_transaction = hash_sha256(hash_sha256(serialized_temp_transaction));

                    // Convert Hex string to char array
                    unsigned char hashed_temp_transaction_bytes[32];
                    hex_string_to_char(hashed_serialized_temp_transaction, hashed_temp_transaction_bytes, 32);
                    unsigned char signature_bytes[current_signature.length() / 2];
                    hex_string_to_char(current_signature, signature_bytes, current_signature.length() / 2);

                    // Get public key
                    string script_public_key = previous_transaction_output.getScriptPubKey();

                    CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey public_key = generate_public_key_from_string(script_public_key);
                    
                    if (!(verify_message(public_key, signature_bytes, current_signature.length() / 2, hashed_temp_transaction_bytes, 32))) {
                        string reason = "";
                        reason += "Signature mismatch\n";
                        reason += "    Signature    : " + current_signature + "\n";
                        reason += "    Public Key   : " + script_public_key + "\n";
                        reason += "    Payload      : " + hashed_serialized_temp_transaction + "\n";

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