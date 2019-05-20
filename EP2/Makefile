CC = g++
CFLAGS = -O -std=c++11
LDFLAGS = -fopenmp
# LINKFLAGS := -Wl,-rpath,$(HOME)/anaconda3/lib
# OSX := -L/usr/local/opt/gcc/lib/gcc/9

OBJS = mandelbrot.cpp

mandelbrot: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -c $(OBJS) `libpng-config --cflags`
	$(CC) $(CFLAGS) $(LDFLAGS) $(OSX) -o mbrot  mandelbrot.o `libpng-config --ldflags`

mandelbrot.o: mandelbrot.cpp

clean:
	rm -f *~ *.o *.png mbrot

test:
#mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>
	./mbrot 0.27085 0.004640 0.27100 0.004810 1000 1000 cpu 10 mb.png
