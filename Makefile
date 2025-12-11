NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = sgemm
OBJ	        = main.o support.o

default: $(EXE)

main.o: main.cu kernel.cu kernel_CPU.c benchmark.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

## I don't think that the followings .o are needed, depending of what we choose to do I will remove them
benchmark.o : benchmark.cu						
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

simulation.o : simulation.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
