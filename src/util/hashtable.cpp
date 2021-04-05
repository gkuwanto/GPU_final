#include <fstream>
#include "hashtable.hpp"
using namespace std;

unsigned int hash_func(const char *key, const unsigned int table_size)
{
	unsigned int h = 0;
	unsigned int o = 31415;
	const unsigned int t = 27183;
	while (*key)
	{
		h = (o * h + *key++) % table_size;
		o = o * t % (table_size - 1);
	}
	return h;
}

// ##################### Hashtable ###########################
Hashtable::Hashtable() : cur_table_item(nullptr), cur_index(0)
{
	table = new HashtableItem*[TABLE_SIZE];
	for (int i = 0; i < TABLE_SIZE; table[i++] = nullptr);
}
Hashtable::~Hashtable()
{
	for (int i = 0; i < TABLE_SIZE; i++)
	{
		if (table[i])
			delete table[i];
	}
	delete []table;
}
bool Hashtable::Add(const string &key, const string &value)
{
	unsigned int i = hash_func(key.c_str(), TABLE_SIZE);
	if (table[i])
	{
		HashtableItem *node;
		for (node = table[i]; node->pnext && (node->pnext->Key() != key); node = node->pnext);
		if (node->pnext)
			return false;
		node->pnext = new HashtableItem(key, value);
		return true;
	}
	table[i] = new HashtableItem(key, value);
	return true;
}
HashtableItem *Hashtable::operator[](const string &key) const
{
	unsigned int i = hash_func(key.c_str(), TABLE_SIZE);
	if (table[i])
	{
		if (table[i]->Key() == key)
			return table[i];
		HashtableItem *node;
		for (node = table[i]; node->pnext && (node->pnext->Key() != key); node = node->pnext);
		if (node->pnext)
			return node->pnext;
	}
	return nullptr;
}
void Hashtable::Remove(const string &key)
{
	unsigned int i = hash_func(key.c_str(), TABLE_SIZE);
	if (table[i])
	{
		HashtableItem *node, *tmp;
		if (table[i]->Key() == key)
		{
			tmp = table[i]->pnext;
			table[i]->pnext = nullptr;
			delete table[i];
			table[i] = tmp;
		}
		else
		{
			for (node = table[i]; node->pnext && (node->pnext->Key() != key); node = node->pnext);
			if (node->pnext) // then key found in linked list
			{
				tmp = node->pnext->pnext;
				node->pnext = nullptr;
				delete node->pnext;
				node->pnext = tmp;
			}
		}
	}
}
void Hashtable::Clear()
{
	for (int i = 0; i < TABLE_SIZE; i++)
	{
		delete table[i];
		table[i] = nullptr;
	}
}
HashtableItem *Hashtable::GetFirst()
{
	int i;
	this->cur_table_item = nullptr;
	this->cur_index = 0;

	for (i = this->cur_index; i < TABLE_SIZE && table[i] == nullptr; i++);
	if (i < TABLE_SIZE)
	{
		this->cur_table_item = table[i];
		this->cur_index = i;
	}
	return this->cur_table_item;
}
HashtableItem *Hashtable::GetNext()
{
	if (this->cur_table_item && this->cur_table_item->pnext)
		this->cur_table_item = this->cur_table_item->pnext;
	else
	{
		int i;
		for (i = this->cur_index + 1; i < TABLE_SIZE && table[i] == nullptr; i++);
		if (i < TABLE_SIZE)
		{
			this->cur_table_item = table[i];
			this->cur_index = i;
		}
		else
		{
			this->cur_table_item = nullptr;
			this->cur_index = 0;
		}
	}
	return this->cur_table_item;
}

void Hashtable::saveHashTable(ofstream& os) {
    HashtableItem *item;
    item = this->GetFirst();
    while(item) {
        os << item->Key() << '\n';
        os << item->Value() << '\n';
        item = this->GetNext();
    }
}

void Hashtable::readHashTable(ifstream& is) {
    string key;
    string value;
    int counter = 0;
    while (is >> key >> value) {
        this->Add(key, value);
        counter++;
    }
}

// ##################### HashtableItem ###########################
HashtableItem::HashtableItem(const string &xKey, const string &xValue) : key(xKey), value(xValue), pnext(nullptr) {}
HashtableItem::~HashtableItem()
{
	if (this->pnext)
		delete this->pnext;
}
const string &HashtableItem::Key() const
{
	return this->key;
}
const string &HashtableItem::Value() const
{
	return this->value;
}
const string &HashtableItem::operator=(const string &value)
{
	this->value = value;
	return value;
}
const char *HashtableItem::operator=(const char *value)
{
	this->value = value;
	return value;
}

