CC = gcc
CFLAGS = -Wall -O2 -pg  # Added -pg for profiling

EXE = nn.exe
SRC = nn.c

all: $(EXE) run profile

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	gprof $(EXE) gmon.out > profile.txt
	cat profile.txt  # Display profiling results

clean:
	rm -f $(EXE) gmon.out profile.txt  # Remove profiling files