__global__ void map_tile(uchar4* d_frame, int w, int h, int maxShift) {
    if (blockIdx.y > h || blockIdx.x > w) return; // tile bounds
    uchar4* outPx = d_frame + blockIdx.y * w * 2 + blockIdx.x + w;
    int offset = (*outPx).x * maxShift / 256;
    uchar4* minInPx = d_frame + (blockIdx.y * w * 2);
    uchar4* maxInPx = d_frame + (blockIdx.y * w * 2 + w - 1);
    uchar4* inPx = outPx - w + offset;
    if (inPx < minInPx) inPx = minInPx;
    if (inPx > maxInPx) inPx = maxInPx;
    *outPx = *inPx;
}

int main() {
    // TODO: load video & copy to GPU
    // TODO: NVDEC frames
    // TODO: for tiles: map_tile
}
