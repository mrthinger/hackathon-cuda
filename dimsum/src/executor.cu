#include "executor.hpp"
#include <vector>

__global__ void len(int* a, int* l)
{}

__global__ void dot(int* a, int* b, int* ret)
{}

__global__ void cossim(int* a, int* b, int* sim)
{
    *sim = 0; // TODO
}

// TODO: basis vec & current lowest sim+key
QueryExecutor::QueryExecutor() {
    // TODO: cudaMalloc
}
QueryExecutor::~QueryExecutor() {
    // TODO: cudaFree
}
void QueryExecutor::processBatch(std::vector<std::vector<float>> batch) {
    // TODO: launch cossim for each
}
