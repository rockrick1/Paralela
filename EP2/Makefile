CC = g++
NVCC=nvcc
CFLAGS = -O -std=c++11
LDFLAGS = -fopenmp
REAL ?= double
CCFLAGS_CUDA=-g -G -DREAL=$(REAL) -arch=sm_30 -Xcompiler -fopenmp
IMGSIZE = 4096
# LINKFLAGS := -Wl,-rpath,$(HOME)/anaconda3/lib
# OSX := -L/usr/local/opt/gcc/lib/gcc/9

OBJS = mandelbrot.cpp
OBJS_CUDA = mandelbrot.cu

mandelbrot:
	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) -c $(OBJS_CUDA) `libpng-config --cflags`
	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) $(OSX) -o mbrot mandelbrot.o -lgomp `libpng-config --ldflags`

mandelbrot.o: mandelbrot.cu

clean:
	rm -f *~ *.o *.png mbrot mbrotgpu

test:
#mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>
	# ./mbrot 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) cpu 1 mb1.png
	./mbrot 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) cpu 10 mb2.png
	./mbrot 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) gpu 128 mb3.png

testgpu:
#mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>
	./mbrotgpu 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) gpu 10 mbgpu.png

.PHONY: gpu
gpu:
	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) -c $(OBJS_CUDA) `libpng-config --cflags`
	$(NVCC) $(CFLAGS) $(CCFLAGS_CUDA) $(OSX) -o mbrotgpu  mandelbrot.o `libpng-config --ldflags`
	# ./mbrotgpu 0.27085 0.004640 0.27100 0.004810 $(IMGSIZE) $(IMGSIZE) gpu 10 mbgpu.png
