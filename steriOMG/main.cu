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
    // Open the video file
    cv::VideoCapture cap("./fixed.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int fps = cap.get(cv::CAP_PROP_FPS);

    // Create a VideoWriter object to save the processed video
    cv::VideoWriter writer("processed_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    // Allocate GPU memory for multiple frames
    const int numFrames = 10; // Process 10 frames per kernel launch
    uchar4 *d_frames;
    size_t frameSize = width * height * sizeof(uchar4);
    cudaMalloc(&d_frames, frameSize * numFrames);

    // Allocate host memory for multiple frames
    std::vector<cv::Mat> frames(numFrames);
    std::vector<cv::Mat> framesRGBA(numFrames);

    // Process frames in batches
    for (int frameStart = 0; frameStart < totalFrames; frameStart += numFrames)
    {
        int framesInBatch = std::min(numFrames, totalFrames - frameStart);

        // Read frames
        for (int i = 0; i < framesInBatch; i++)
        {
            cap >> frames[i];
            if (frames[i].empty())
                break;
            cv::cvtColor(frames[i], framesRGBA[i], cv::COLOR_BGR2RGBA);
        }

        // Copy frames to GPU
        for (int i = 0; i < framesInBatch; i++)
        {
            cudaMemcpy(d_frames + i * width * height, framesRGBA[i].data, frameSize, cudaMemcpyHostToDevice);
        }

        // Call map_tile kernel for each frame
        dim3 blockSize(1, 1);
        dim3 gridSize(width / 2, height);
        int maxShift = 32; // Adjust this value as needed
        for (int i = 0; i < framesInBatch; i++)
        {
            map_tile<<<gridSize, blockSize>>>(d_frames + i * width * height, width / 2, height, maxShift);
        }

        // Copy results back to CPU
        for (int i = 0; i < framesInBatch; i++)
        {
            cudaMemcpy(framesRGBA[i].data, d_frames + i * width * height, frameSize, cudaMemcpyDeviceToHost);
        }

        // Convert back to BGR and write to output video
        for (int i = 0; i < framesInBatch; i++)
        {
            cv::Mat outputFrame;
            cv::cvtColor(framesRGBA[i], outputFrame, cv::COLOR_RGBA2BGR);
            writer.write(outputFrame);
        }
    }

    // Clean up
    cudaFree(d_frames);
    cap.release();
    writer.release();

    return 0;
}