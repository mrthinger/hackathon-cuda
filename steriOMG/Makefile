FFMPEG_INCLUDE = $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale)
FFMPEG_LIBS = $(shell pkg-config --libs libavcodec libavformat libavutil libswscale)

steriOMG: main.cu
	nvcc $(FFMPEG_INCLUDE) main.cu -o steriOMG $(FFMPEG_LIBS) -Xlinker -lm -Xlinker -ldl

clean:
	rm -f steriOMG

install-deps:
	sudo apt-get update
	sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev