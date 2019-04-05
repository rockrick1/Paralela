CC = g++
CFLAGS = -O -pthread -std=c++17

OBJS = pgrep.cpp

pgrep: $(OBJS)
	$(CC) $(CFLAGS) -o pgrep $(OBJS)

pgrep.o: pgrep.cpp

clean:
	rm -f *~ *.o pgrep