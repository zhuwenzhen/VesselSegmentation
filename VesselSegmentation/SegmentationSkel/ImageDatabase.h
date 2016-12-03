#ifndef IMAGEDATABASE_H
#define IMAGEDATABASE_H

#include <string>
#include <vector>


using namespace std;

// A DatabaseItem holds the name of an image, 
// The images themselves are not stored in memory.
struct DatabaseItem {
	string name;
};

// The ImageDatabase class is a vector of database items.
class ImageDatabase : public vector<DatabaseItem> {
public:
	// Create a new database.
	ImageDatabase();

	// Load a database from file.
	bool load(const char *name, bool sift);
};

#endif