// From : http://www.prowaretech.com/Computer/Cpp/HashtableString
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

#define TABLE_SIZE 1000003 // use a large prime number


unsigned int hash_func(const char *key, const unsigned int table_size);

class HashtableItem;

class Hashtable
{
private:
	HashtableItem **table;
	HashtableItem *cur_table_item;
	int cur_index;

public:
	Hashtable();
	~Hashtable();

	// Add a new entry, returns false when the key already exists
	bool Add(const string &key, const string &value); 

	HashtableItem *operator[](const string &key) const;

	 // removes one table entry
	void Remove(const string &key);

	 // removes all the table entries
	void Clear();

	// for looping through the table of kes/values
	HashtableItem *GetFirst();
	HashtableItem *GetNext();

    // for file saving
    void saveHashTable(ofstream&);
    void readHashTable(ifstream&);
};

class HashtableItem
{
private:
	HashtableItem *pnext;
	string key, value;

	// keep these private to prevent the client from creating this object
	HashtableItem(){}
	HashtableItem(const string &key, const string &value);
	~HashtableItem();

public:
	const string &Key() const;
	const string &Value() const;
	const string &operator=(const string &value);
	const char *operator=(const char *value);

	// some friend functions that can access the private data
	friend bool Hashtable::Add(const string &key, const string &value);
	friend Hashtable::~Hashtable();
	friend void Hashtable::Remove(const string &key);
	friend HashtableItem *Hashtable::operator[](const string &key) const;
	friend HashtableItem *Hashtable::GetNext();
	friend void Hashtable::Clear();
};
