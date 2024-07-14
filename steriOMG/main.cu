#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

const int SHIFT_SIZE = 32;

__global__ void map_frames(uchar4 *d_frames, int w, int h)
{
    uchar4 *d_frame = d_frames + blockIdx.z * w * h * 2;
    uchar4 *outPx = d_frame + blockIdx.y * w * 2 + blockIdx.x + w;
    uchar4 *rowStart = d_frame + (blockIdx.y * w * 2);
    *outPx = *(max(rowStart, min(rowStart + w - 1, outPx - w - ((*outPx).x * SHIFT_SIZE >> 8 /* >>8=/256*/))));
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

    int sbsWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter writer("processed_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cap.get(cv::CAP_PROP_FPS), cv::Size(width, height));

    const int batchSize = 32;

    uchar4 *d_frames;
    size_t frameSize = sbsWidth * height * sizeof(uchar4);
    cudaMalloc(&d_frames, frameSize * batchSize);

    std::vector<cv::Mat> frames(batchSize);
    std::vector<cv::Mat> framesRGBA(batchSize);

    for (int batchStart = 0; batchStart < totalFrames; batchStart += batchSize)
    {
        auto start = std::chrono::high_resolution_clock::now();
        int framesInBatch = std::min(batchSize, totalFrames - batchStart);
        for (int i = 0; i < framesInBatch; i++)
        {
            cap >> frames[i];
            if (frames[i].empty())
                break;
            cv::cvtColor(frames[i], framesRGBA[i], cv::COLOR_BGR2RGBA);
            cudaMemcpy(d_frames + i * width * height, framesRGBA[i].data, frameSize, cudaMemcpyHostToDevice);
        }

        dim3 gridSize(sbsWidth / 2, height, framesInBatch);
        map_frames<<<gridSize, 1>>>(d_frames, sbsWidth / 2, height);
        cudaDeviceSynchronize();

        for (int i = 0; i < framesInBatch; i++)
        {
            cudaMemcpy(framesRGBA[i].data, d_frames + i * width * height, frameSize, cudaMemcpyDeviceToHost);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double fps = framesInBatch / elapsed.count();
        std::cout << "Kernel execution time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Frames per second: " << fps << std::endl;

        // Convert back to BGR and write to output video
        for (int i = 0; i < framesInBatch; i++)
        {
            cv::Mat outputFrame;
            cv::cvtColor(framesRGBA[i], outputFrame, cv::COLOR_RGBA2BGR);
            writer.write(outputFrame);
        }
    }

    cudaFree(d_frames);
    cap.release();
    writer.release();
    return 0;
}
