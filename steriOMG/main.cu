#include <cuda_runtime.h>
#include "nvEncodeAPI.h"
#include "nvcuvid.h"
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                          \
    {                                                             \
        CUresult result = call;                                   \
        if (result != CUDA_SUCCESS)                               \
        {                                                         \
            const char *err_name;                                 \
            cuGetErrorName(result, &err_name);                    \
            std::cerr << "CUDA error: " << err_name << std::endl; \
            exit(1);                                              \
        }                                                         \
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

// Global variables for NVDEC
CUcontext g_ctx = NULL;
CUvideodecoder g_decoder = NULL;
std::vector<uchar4 *> g_frameBuffer;
int g_frameWidth = 0, g_frameHeight = 0;

// Callback functions for NVDEC
static int CUDAAPI HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pVideoFormat)
{
    g_frameWidth = pVideoFormat->coded_width;
    g_frameHeight = pVideoFormat->coded_height;
    return 1;
}

static int CUDAAPI HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams)
{
    return 1;
}

static int CUDAAPI HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo)
{
    CUdeviceptr dpSrcFrame = 0;
    unsigned int nSrcPitch = 0;
    CUVIDPROCPARAMS oVPP = {0};
    oVPP.progressive_frame = pDispInfo->progressive_frame;
    oVPP.second_field = pDispInfo->repeat_first_field + 1;
    oVPP.top_field_first = pDispInfo->top_field_first;
    oVPP.unpaired_field = pDispInfo->repeat_first_field < 0;
    oVPP.output_stream = 0;

    CUDA_CHECK(cuvidMapVideoFrame(g_decoder, pDispInfo->picture_index, &dpSrcFrame, &nSrcPitch, &oVPP));

    uchar4 *d_frame;
    CUDA_CHECK(cuMemAlloc((CUdeviceptr *)&d_frame, g_frameWidth * g_frameHeight * sizeof(uchar4)));

    // Convert NV12 to RGBA
    // Note: This is a placeholder. You need to implement the actual conversion.
    // cudaConvertNV12toRGBA(dpSrcFrame, nSrcPitch, d_frame, g_frameWidth * sizeof(uchar4), g_frameWidth, g_frameHeight);

    g_frameBuffer.push_back(d_frame);

    CUDA_CHECK(cuvidUnmapVideoFrame(g_decoder, dpSrcFrame));
    return 1;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return -1;
    }

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    // Initialize CUDA
    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuCtxCreate(&g_ctx, 0, 0));

    // Open input file
    AVFormatContext *formatContext = nullptr;
    if (avformat_open_input(&formatContext, inputFile, nullptr, nullptr) != 0)
    {
        std::cerr << "Could not open input file" << std::endl;
        return -1;
    }

    // Find video stream
    int videoStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++)
    {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1)
    {
        std::cerr << "Could not find video stream" << std::endl;
        return -1;
    }

    // Set up NVDEC
    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = cudaVideoCodec_H264;
    videoParserParameters.ulMaxNumDecodeSurfaces = 1;
    videoParserParameters.ulClockRate = 0;
    videoParserParameters.ulErrorThreshold = 100;
    videoParserParameters.ulMaxDisplayDelay = 0;
    // videoParserParameters.uVidPicStruct = 0;
    videoParserParameters.pExtVideoInfo = nullptr;
    videoParserParameters.pfnSequenceCallback = HandleVideoSequence;
    videoParserParameters.pfnDecodePicture = HandlePictureDecode;
    videoParserParameters.pfnDisplayPicture = HandlePictureDisplay;

    CUvideoparser videoParser = nullptr;
    CUDA_CHECK(cuvidCreateVideoParser(&videoParser, &videoParserParameters));

    // Main decoding loop
    AVPacket *packet = av_packet_alloc();
    while (av_read_frame(formatContext, packet) >= 0)
    {
        if (packet->stream_index == videoStreamIndex)
        {
            CUVIDSOURCEDATAPACKET cuvidPacket = {0};
            cuvidPacket.payload = packet->data;
            cuvidPacket.payload_size = packet->size;
            cuvidPacket.flags = CUVID_PKT_TIMESTAMP;
            cuvidPacket.timestamp = packet->pts;

            if (packet->flags & AV_PKT_FLAG_KEY)
                cuvidPacket.flags |= CUVID_PKT_ENDOFPICTURE;

            CUDA_CHECK(cuvidParseVideoData(videoParser, &cuvidPacket));
        }
        av_packet_unref(packet);
    }

    // Signal end of stream
    CUVIDSOURCEDATAPACKET cuvidPacket = {0};
    cuvidPacket.flags = CUVID_PKT_ENDOFSTREAM;
    CUDA_CHECK(cuvidParseVideoData(videoParser, &cuvidPacket));

    // Process frames
    for (size_t i = 0; i < g_frameBuffer.size(); ++i)
    {
        uchar4 *d_frame = g_frameBuffer[i];

        // Call map_tile kernel
        dim3 blockSize(1, 1);
        dim3 gridSize(g_frameWidth / 2, g_frameHeight);
        int maxShift = 32; // Adjust this value as needed
        map_tile<<<gridSize, blockSize>>>(d_frame, g_frameWidth / 2, g_frameHeight, maxShift);

        // Here you would typically encode the processed frame
        // For simplicity, we're just freeing the memory
        CUDA_CHECK(cuMemFree((CUdeviceptr)d_frame));
    }

    // Clean up
    av_packet_free(&packet);
    avformat_close_input(&formatContext);
    cuvidDestroyVideoParser(videoParser);
    if (g_decoder)
    {
        cuvidDestroyDecoder(g_decoder);
    }
    cuCtxDestroy(g_ctx);

    return 0;
}