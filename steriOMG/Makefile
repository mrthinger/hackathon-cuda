OPENCV_INCLUDE = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

steriOMG: main.cu
	nvcc $(OPENCV_INCLUDE) main.cu -o steriOMG $(OPENCV_LIBS)

clean:
	rm -f steriOMG