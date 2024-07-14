__global__ void map_tile(uchar4* d_inframe, uchar4* d_outframe, int w, int h) {
    int outIdx = threadIdx.y * w + threadIdx.x;
    d_outframe[outIdx] = d_inframe[outIdx]; // FIXME: offset
}

int main() {
    // TODO: load video & copy to GPU
    // TODO: NVDEC frames
    // TODO: for tiles: map_tile
}
