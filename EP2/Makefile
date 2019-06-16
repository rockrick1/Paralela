CC = g++
NVCC=nvcc
MPICC=mpiCC
CFLAGS = -O -std=c++11
LDFLAGS = -fopenmp -lmpi
REAL ?= float
CCFLAGS_CUDA= -g -G -arch=sm_30 -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi -Xcompiler " -fopenmp -Bsymbolic-functions -z,relro -I/usr/include/mpich -L/usr/lib/x86_64-linux-gnu -lmpichcxx -lmpich -lmpi"
IMGSIZE = 1024
# LINKFLAGS := -Wl,-rpath,$(HOME)/anaconda3/lib
# OSX := -L/usr/local/opt/gcc/lib/gcc/9

OBJS = mandelbrotmpi.cpp
OBJS_CUDA = mandelbrot.cu
OBJS_MPI_CUDA = mandelbrotmpi.cu

mandelbrot: $(OBJS)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -c $(OBJS) `libpng-config --cflags`
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OSX) -o mbrotmpi mandelbrotmpi.o `libpng-config --ldflags`

mandelbrotmpi.o: mandelbrotmpi.cpp

# mandelbrot: $(OBJS)
# 	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) -c $(OBJS_MPI_CUDA) `libpng-config --cflags`
# 	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) $(OSX) -o mbrotgpu  mandelbrotmpi.o `libpng-config --ldflags`
#
# mandelbrotmpi.o: mandelbrotmpi.cu

clean:
	rm -f *~ *.o *.png mbrot mbrotgpu mbrotmpi

test:
#mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>
	./mbrotgpu 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) cpu 10 mb.png

testgpu:
#mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>
	./mbrotgpu 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) gpu 10 mbgpu.png

testmpi:
	mpirun -n 4 ./mbrotgpu 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) gpu 128 mbgpu.png
	mpirun -n 4 ./mbrotgpu 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) seq 1 mbseq.png
	mpirun -n 4 ./mbrotgpu 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) cpu 4 mbcpu.png

.PHONY: gpu
gpu:
	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) -c $(OBJS_MPI_CUDA) `libpng-config --cflags`
	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) $(OSX) -o mbrotgpu  mandelbrotmpi.o `libpng-config --ldflags`

.PHONY: mpi
mpi:
	$(CC) $(CFLAGS) $(LDFLAGS) -c $(OBJS_MPI) `libpng-config --cflags`
	$(CC) $(CFLAGS) $(LDFLAGS) $(OSX) -o mbrotmpi  mandelbrot.o `libpng-config --ldflags`
