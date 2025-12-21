NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -gencode arch=compute_80,code=sm_80
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
SRC         = main.cu kernel.cu benchmark.cu simulation.cu csv.cu
EXE	        = account_savings
OBJ	        = $(SRC:.cu=.o)

all: $(EXE)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

main.o: main.cu kernel.h benchmark.h simulation.h cuda_support.h csv.h
kernel.o: kernel.cu kernel.h
benchmark.o : benchmark.cu benchmark.h
simulation.o : simulation.cu simulation.h
csv.o : csv.cu csv.h

clean:
	rm -rf *.o $(EXE)
	rm -f out_balance.csv
	rm -f out_sums.csv
	rm -f testing.csv
	rm -f core.*
