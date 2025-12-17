NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
SRC         = main.cu kernel.cu benchmark.cu simulation.cu
EXE	        = account_savings
OBJ	        = $(SRC:.cu=.o)

all: $(EXE)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

main.o: main.cu kernel.h benchmark.h simulation.h cuda_support.h
kernel.o: kernel.cu kernel.h
benchmark.o : benchmark.cu benchmark.h
simulation.o : simulation.cu simulation.h

clean:
	rm -rf *.o $(EXE)
