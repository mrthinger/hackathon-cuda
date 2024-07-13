dimsum: src/main.cpp src/executor.o
	nvcc -o dimsum $^

%.o: %.cu
	nvcc -dc -o $@ $<
