// __global__ void shift_video(uint8_t* video_buffer, const int* depth_map, int num_frames, int height, int width, int max_shift) {
//     int frame_idx = blockIdx.z;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int x = blockIdx.x * blockDim.x + threadIdx.x;

//     if (y < height && x < (width / 2)) {
//         int single_video_width = width / 2;

//         // Get the shift value for the current pixel
//         int shift = depth_map[frame_idx * height * single_video_width + y * single_video_width + x];
//         int shifted_x = min(max(x + shift, 0), single_video_width - 1);

//         // Set the shifted pixel value
//         for (int c = 0; c < 3; ++c) {  // Assuming RGB format
//             video_buffer[(frame_idx * height + y) * width * 3 + (single_video_width + x) * 3 + c] =
//                 video_buffer[(frame_idx * height + y) * width * 3 + shifted_x * 3 + c];
//         }
//     }
// }

__global__ void shift_video(uchar4* d_frame, int w, int h, int maxShift) {
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
