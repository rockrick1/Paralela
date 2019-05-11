CC = g++
CFLAGS = -O -std=c++11
# LINKFLAGS := -Wl,-rpath,$(HOME)/anaconda3/lib

OBJS = mandelbrot.cpp

mandelbrot: $(OBJS)
	$(CC) $(CFLAGS) -c $(OBJS) `libpng-config --cflags`
	$(CC) $(CFLAGS) -o mbrot  mandelbrot.o `libpng-config --ldflags`

mandelbrot.o: mandelbrot.cpp

clean:
	rm -f *~ *.o *.png mbrot

test:
	./mbrot 0.27085 0.004640 0.27100 0.004810 1000 1000 cpu 2 mb.png