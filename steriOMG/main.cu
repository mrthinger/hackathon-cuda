#include <cuda_runtime.h>
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

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
    // FFmpeg initialization
    AVFormatContext *formatContext = NULL;
    AVCodecContext *codecContext = NULL;
    AVFrame *frame = NULL;
    AVPacket *packet = NULL;
    int videoStreamIndex = -1;

    // Open input file
    if (avformat_open_input(&formatContext, "fixed.mp4", NULL, NULL) != 0)
    {
        fprintf(stderr, "Could not open input file\n");
        return -1;
    }

    // Find video stream
    for (int i = 0; i < formatContext->nb_streams; i++)
    {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1)
    {
        fprintf(stderr, "Could not find video stream\n");
        return -1;
    }

    // Set up codec context
    const AVCodec *codec = avcodec_find_decoder(formatContext->streams[videoStreamIndex]->codecpar->codec_id);
    codecContext = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecContext, formatContext->streams[videoStreamIndex]->codecpar);
    avcodec_open2(codecContext, codec, NULL);

    // Allocate frame and packet
    frame = av_frame_alloc();
    packet = av_packet_alloc();

    // Allocate GPU memory
    uchar4 *d_frame;
    int width = codecContext->width;
    int height = codecContext->height;
    size_t frameSize = width * height * sizeof(uchar4);
    cudaMalloc(&d_frame, frameSize);

    // Main loop for reading frames
    while (av_read_frame(formatContext, packet) >= 0)
    {
        if (packet->stream_index == videoStreamIndex)
        {
            int response = avcodec_send_packet(codecContext, packet);
            if (response < 0)
            {
                fprintf(stderr, "Error sending packet for decoding\n");
                break;
            }

            while (response >= 0)
            {
                response = avcodec_receive_frame(codecContext, frame);
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF)
                {
                    break;
                }
                else if (response < 0)
                {
                    fprintf(stderr, "Error during decoding\n");
                    break;
                }

                // Convert frame to RGBA and copy to GPU
                // You'll need to implement this conversion
                // convertFrameToRGBA(frame, d_frame, width, height);

                // Call map_tile kernel
                dim3 blockSize(1, 1);
                dim3 gridSize(width / 2, height);
                int maxShift = 32; // Adjust this value as needed
                map_tile<<<gridSize, blockSize>>>(d_frame, width / 2, height, maxShift);

                // Process the frame (e.g., display or save)
                // You'll need to implement this part
                // processFrame(d_frame, width, height);
            }
        }
        av_packet_unref(packet);
    }

    // Clean up
    cudaFree(d_frame);
    avcodec_free_context(&codecContext);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avformat_close_input(&formatContext);

    return 0;
}
