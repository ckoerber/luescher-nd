CC=g++-9

CXXFLAGS += -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic \
            -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual \
            -Wconversion -Wsign-conversion -Wnull-dereference \
            -Wdouble-promotion -Wformat=2

# gcc only
CXXFLAGS += -Wmisleading-indentation \
            -Wduplicated-cond -Wduplicated-branches -Wlogical-op \
            -Wuseless-cast -Wdouble-promotion -Wformat=2

zeta: main.cc zeta.o
	$(CC) $(CXXFLAGS) main.cc -o zeta zeta.o

zeta.o: zeta.h zeta.cc
	$(CC) $(CXXFLAGS) -c zeta.cc

clean:
	$(RM) zeta.o zeta
