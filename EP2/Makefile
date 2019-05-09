CC = g++
CFLAGS = -O -std=c++11

OBJS = mandelbrot.cpp

mandelbrot: $(OBJS)
	$(CC) $(CFLAGS) -o mandelbrot $(OBJS)

mandelbrot.o: mandelbrot.cpp

clean:
	rm -f *~ *.o mandelbrot