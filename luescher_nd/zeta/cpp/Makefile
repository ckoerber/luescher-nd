#CC=g++
CC=clang++ --std=c++17

CXXFLAGS += -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic \
            -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual \
            -Wconversion -Wsign-conversion -Wnull-dereference \
            -Wdouble-promotion -Wformat=2

# gcc only
CXXFLAGS += -Wmisleading-indentation \
            -Wduplicated-cond -Wduplicated-branches -Wlogical-op \
            -Wuseless-cast -Wdouble-promotion -Wformat=2

# make CUBATURE=path/to/cubature
ifdef CUBATURE
CUBATURE_FILE=$(CUBATURE)/hcubature.c
CUBATURE_INCLUDE=-I$(CUBATURE)
CUBATURE_FLAG=-DCUBATURE
endif

zeta: main.cc zeta.o zeta_flat.h
	$(CC) $(CXXFLAGS) $(CUBATURE_FLAG) main.cc -o zeta zeta.o $(CUBATURE_FILE) $(CUBATURE_INCLUDE)

zeta.o: zeta.h zeta.cc
	$(CC) $(CXXFLAGS) $(CUBATURE_FLAG) -c zeta.cc $(CUBATURE_FILE) $(CUBATURE_INCLUDE)

clean:
	$(RM) zeta.o zeta
