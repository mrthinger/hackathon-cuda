#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include <ffmpeg/ffmpeg.h> // Replace with actual FFmpeg header files
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp> // Replace with actual OpenCV headers

void process_video(const std::string& input_path, const std::string& output_path, int max_shift) {
    VideoInfo video_info = get_video_info(input_path);
    int output_width = video_info.width;

    const size_t CHUNK_SIZE = 5 * (1024 * 1024 * 1024);  // 5 GB
    size_t CHUNK_FRAMES = CHUNK_SIZE / (output_width * video_info.height * 3);

    cudaSetDevice(0);
    
    std::vector<uint8_t> vbuffer(CHUNK_FRAMES * video_info.height * output_width * 3);
    
    // Create buffers for input/output streams
    auto process_input = ffmpeg::input(input_path)
        .output("pipe:", "rawvideo", "rgb24")
        .run_async();
    
    auto in_modified = ffmpeg::input("pipe:")
        .format("rawvideo")
        .pix_fmt("rgb24")
        .size(std::to_string(output_width) + "x" + std::to_string(video_info.height))
        .framerate(video_info.framerate)
        .hwaccel("cuda");

    auto in_original = ffmpeg::input(input_path)
        .hwaccel("cuda");

    auto process_output = ffmpeg::output(in_modified, in_original, output_path)
        .acodec("copy")
        .framerate(video_info.framerate)
        .size(std::to_string(output_width) + "x" + std::to_string(video_info.height))
        .overwrite_output()
        .run_async();

    for (size_t chunk_start = 0; chunk_start < video_info.num_frames; chunk_start += CHUNK_FRAMES) {
        size_t chunk_end = std::min(chunk_start + CHUNK_FRAMES, video_info.num_frames);
        size_t chunk_frames = chunk_end - chunk_start;

        for (size_t i = 0; i < chunk_frames; ++i) {
            std::vector<uint8_t> in_bytes(video_info.width * video_info.height * 3);
            if (!process_input.read(in_bytes.data(), in_bytes.size())) break;

            cv::Mat data_gpu(video_info.height, video_info.width, CV_8UC3, in_bytes.data());
            std::memcpy(vbuffer.data() + i * video_info.height * output_width * 3, data_gpu.data, data_gpu.total() * data_gpu.elemSize());
        }

        for (size_t i = 0; i < chunk_frames; ++i) {
            process_output.stdin.write(reinterpret_cast<char*>(vbuffer.data() + i * video_info.height * output_width * 3), video_info.height * output_width * 3);
        }
    }

    process_output.stdin.close();
    process_output.wait();
}


int main(int argc, char* argv[]) {
    std::filesystem::create_directories("./build/sbs");

    int max_shift = 21;
    std::string filename = (argc > 1) ? argv[1] : "test.mp4";
    
    std::string filepath = "./build/depth/" + filename;
    std::string output_path = "./build/sbs/88qv-" + std::to_string(max_shift) + "-" + filename;

    process_video(filepath, output_path, max_shift);
    return 0;
}