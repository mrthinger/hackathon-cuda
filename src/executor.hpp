#include <string>
#include <vector>
// TODO: use https://github.com/minhhn2910/cuda-half2

class QueryExecutor {
public:
    std::vector<float> basis; // FIXME: take this type in constructor then cudaMalloc & cudaMemcpy
    std::string curkey;
    float cursim;
    QueryExecutor();
    ~QueryExecutor();
    void processBatch(std::vector<std::vector<float>> batch);
};
