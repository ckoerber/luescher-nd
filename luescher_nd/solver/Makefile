CXX = clang++
CXXFLAGS = -Wall -g -O3
INC = -I/usr/local/include/eigen3 -I/Users/christopherkorber/local/spectra/include

OBJECTS = solver.o

test: $(OBJECTS)
	$(CXX) -o $@ solver.o

%.o: %.cpp $(HDRS)
	$(CXX) $(INC) $(CXXFLAGS) -c $<

.PHONY: clean
clean:
	$(RM) $(OBJECTS)
	$(RM) -f test
