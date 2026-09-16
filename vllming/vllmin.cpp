#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <utility>

using namespace std;


int main(){
    ifstream safetensors_file("model.safetensors", ios_base::binary);
    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
}







