CC = g++
CFLAGS = -O -std=c++11 -lpng -lz
# LINKFLAGS := -Wl,-rpath,$(HOME)/anaconda3/lib

OBJS = mandelbrot.cpp

mandelbrot: $(OBJS)
	$(CC) $(CFLAGS) -c $(OBJS) `libpng-config --cflags`
	$(CC) $(CFLAGS) -o mandelbrot  mandelbrot.o `libpng-config --ldflags`

mandelbrot.o: mandelbrot.cpp

clean:
	rm -f *~ *.o #mandelbrot