CC = g++
CFLAGS = -O -std=c++11

OBJS = mandelbrot.cpp

mandelbrot: $(OBJS)
	$(CC) $(CFLAGS) -c $(OBJS) `libpng-config --cflags` 
	$(CC) $(CFLAGS) -o mandelbrot  mandelbrot.o `libpng-config --ldflags` 

mandelbrot.o: mandelbrot.cpp

clean:
	rm -f *~ *.o mandelbrot