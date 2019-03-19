CC=gcc
CFLAGS=-Wall -pedantic -O2
OBJECTS=pgrep.c #stringReader.o

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<

pgrep: $(OBJECTS)
	$(CC) $(OBJECTS) $(CFLAGS) -o pgrep

clean:
	rm -rf *.o