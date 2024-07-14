FFMPEG_INCLUDE = $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale)
FFMPEG_LIBS = $(shell pkg-config --libs libavcodec libavformat libavutil libswscale)
NVCODEC_PATH = /home/evanp/hackathon-cuda/Video_Codec_SDK_12.2.72
NVCODEC_OBJS = $(NVCODEC_PATH)/Lib/linux/stubs/x86_64/libnvcuvid.so $(NVCODEC_PATH)/Lib/linux/stubs/x86_64/libnvidia-encode.so
CUDA_LIBS = -lcuda -lcudart

steriOMG: main.cu
	nvcc $(FFMPEG_INCLUDE) main.cu -o steriOMG $(FFMPEG_LIBS) $(NVCODEC_OBJS) $(CUDA_LIBS) -Xlinker -lm -Xlinker -ldl

# ... rest of the Makefile remains unchanged ...
clean:
	rm -f steriOMG

install-deps:
	sudo apt-get update
	sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev