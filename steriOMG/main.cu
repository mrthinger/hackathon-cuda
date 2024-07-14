#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void map_tile(uchar4 *d_frame, int w, int h, int maxShift)
{
    if (blockIdx.y > h || blockIdx.x > w)
        return; // tile bounds
    uchar4 *outPx = d_frame + blockIdx.y * w * 2 + blockIdx.x + w;
    int offset = (*outPx).x * maxShift / 256;
    uchar4 *minInPx = d_frame + (blockIdx.y * w * 2);
    uchar4 *maxInPx = d_frame + (blockIdx.y * w * 2 + w - 1);
    uchar4 *inPx = outPx - w - offset;
    if (inPx < minInPx)
        inPx = minInPx;
    if (inPx > maxInPx)
        inPx = maxInPx;
    *outPx = *inPx;
}

int main()
{
    // Load image using OpenCV
    cv::Mat frame = cv::imread("./first_frame.jpg", cv::IMREAD_COLOR);
    if (frame.empty())
    {
        std::cerr << "Error: Could not read the image file." << std::endl;
        return -1;
    }

    // Convert to RGBA
    cv::Mat frameRGBA;
    cv::cvtColor(frame, frameRGBA, cv::COLOR_BGR2RGBA);

    // Allocate GPU memory
    uchar4 *d_frame;
    int width = frameRGBA.cols;
    int height = frameRGBA.rows;
    size_t frameSize = width * height * sizeof(uchar4);
    cudaMalloc(&d_frame, frameSize);

    // Copy image data to GPU
    cudaMemcpy(d_frame, frameRGBA.data, frameSize, cudaMemcpyHostToDevice);

    // Call map_tile kernel
    dim3 blockSize(1, 1);
    dim3 gridSize(width / 2, height);
    int maxShift = 32; // Adjust this value as needed
    map_tile<<<gridSize, blockSize>>>(d_frame, width / 2, height, maxShift);

    // Copy result back to CPU
    cudaMemcpy(frameRGBA.data, d_frame, frameSize, cudaMemcpyDeviceToHost);

    // Convert back to BGR for saving
    cv::Mat outputFrame;
    cv::cvtColor(frameRGBA, outputFrame, cv::COLOR_RGBA2BGR);

    // Save the processed image
    cv::imwrite("processed_frame.jpg", outputFrame);

    // Clean up
    cudaFree(d_frame);

    return 0;
}
