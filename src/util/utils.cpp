#include <boost/algorithm/hex.hpp>
#include <crypto++/dsa.h>
#include <crypto++/hex.h>
#include <crypto++/oids.h>
#include <iomanip>
#include <random>
#include <sstream>
#include "utils.hpp"

using namespace std;

/*  Parses VARINT element of a transaction.
 *
 *  We need the address of prefix and size because we want the change to persist outside of the function
 *  Same as offset, we pass the address to make the change the persistence. In short, we want to pass by reference, not value.
 */
void parse_variable_integer(string raw_transaction, string& prefix, string& size, int* offset) {
    prefix = raw_transaction.substr(*offset, 1);
    (*offset)++;

    if (prefix == "FD") {
        size = raw_transaction.substr(*offset, 2);
        prefix = "FD";
        (*offset) += 2;
    } else if (prefix == "FE") {
        size = raw_transaction.substr(*offset, 4);
        prefix = "FE";
        (*offset) += 4;
    } else if (prefix == "FF") {
        size = raw_transaction.substr(*offset, 8);
        prefix = "FF";
        (*offset) += 8;
    } else {
        size = prefix;
        prefix = "";
    }
}


/*  Converts a float into transaction-compliant 8 bytes hexadecimal value
 *
 *  Per Bitcoin standard, the value of float must be multiplied by 10^8 before casted into 8 bytes hexadecimal
 */
string float_to_long_hex(float value) {
    stringstream stream;
    stream << setfill('0') << setw(sizeof(unsigned long long int)*2) << hex << (long long int) value * 100000000;
    
    return stream.str();
}

/*  Converts hexadecimal string to 8 bytes long integer  */ 
long long int hex_string_to_long(string hex_string) {
    const char* temp_hex_string = hex_string.c_str();
    stringstream ss;
    ss << hex;

    for (unsigned int i = 0; i < hex_string.length(); i++) {
        ss << (unsigned int) temp_hex_string[i];
    }

    long long int result;
    ss >> result;
    return result;
}

/*  Converts integer to VARINT element of transaction
 *  As before, we want changes to prefix and size to persist, so we pass them by reference
 */
void variable_int_to_hex_string(int value, string& prefix, string& size) {
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

/*  Flips hex string endianess  */
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

/*  Converts hex string to respective char  */
void hex_string_to_char(string& value, unsigned char* value_in_char_array, int len) {
    stringstream value_stream;
    vector<unsigned char> value_in_bytes;

    unsigned int c;
    unsigned int offset = 0;
    while (offset < value.length()) {
        value_stream.clear();
        value_stream << hex << value.substr(offset, 2);
        value_stream >> c;

        value_in_bytes.push_back((unsigned char) c);
        offset += 2;
    }

    unsigned char* temp_value_in_char_array = value_in_bytes.data();
    memmove(value_in_char_array, temp_value_in_char_array, len);
}


/*  Generate Bitcoin accounts   */
vector<Account> generate_accounts(int n) {
    vector<Account> accounts;

    for (int i = 0; i < n; i++) {
        accounts.push_back(Account());
    }

    return accounts;
}


/*  Hashes payload with SHA256 hash algorithm   */
string hash_sha256(const string& payload) {
    unsigned char digest[CryptoPP::SHA256::DIGESTSIZE];
    CryptoPP::SHA256().CalculateDigest(digest, (unsigned char *) payload.c_str(), payload.size());

    string hashed_payload;
    CryptoPP::HexEncoder encoder;
    encoder.Attach(new CryptoPP::StringSink(hashed_payload));
    encoder.Put(digest, sizeof(digest));
    encoder.MessageEnd();

    return hashed_payload;
}

/*  Generates a public key object from public key string  */
CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey generate_public_key_from_string(string public_key) {
    CryptoPP::HexDecoder decoder;
    decoder.Put((unsigned char*) &public_key[0], public_key.size());
    decoder.MessageEnd();

    CryptoPP::ECP::Point elliptic_curve_points;
    size_t length = decoder.MaxRetrievable();

    elliptic_curve_points.identity = false;
    elliptic_curve_points.x.Decode(decoder, length/2);
    elliptic_curve_points.y.Decode(decoder, length/2);

    CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey encoded_public_key;
    encoded_public_key.Initialize(CryptoPP::ASN1::secp256k1(), elliptic_curve_points);
    
    return encoded_public_key;
}

/*  Verifies a message with the provided signature and public key  */
bool verify_message(CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::PublicKey& public_key, unsigned char* signature, int signature_len, unsigned char* message, int message_len) {
    CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA256>::Verifier verifier(public_key);

    // Convert DER to P1363
    unsigned char p1363_signature[verifier.SignatureLength()];
    size_t encoded_size = CryptoPP::DSAConvertSignatureFormat(p1363_signature, verifier.SignatureLength(), CryptoPP::DSA_P1363, signature, signature_len, CryptoPP::DSA_DER);
    string p1363_signature_string((char *) p1363_signature, verifier.SignatureLength());
    p1363_signature_string.resize(encoded_size);

    bool result = verifier.VerifyMessage(message, message_len, (unsigned char *) p1363_signature_string.data(), p1363_signature_string.size());

    return result;
}