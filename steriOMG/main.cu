__global__ void map_tile(uchar4* d_inframe, uchar4* d_outframe, int w, int h, int xoffset, int yoffset) {
    if (threadIdx.y + yoffset > h || threadIdx.x + xoffset > w) return; // tile bounds
    int yOutCoord = threadIdx.y + yoffset;
    int xOutCoord = threadIdx.x + xoffset;
    int outIdx = (threadIdx.y + yoffset) * w + (threadIdx.x + xoffset); // FIXME: skip right half of frame
    int inIdx = outIdx; // TODO: horizontal offset
    d_outframe[outIdx] = d_inframe[inIdx];
}

int main() {
    // TODO: load video & copy to GPU
    // TODO: NVDEC frames
    // TODO: for tiles: map_tile
}
