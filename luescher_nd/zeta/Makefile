CXX = g++
CXXFLAGS = --std=c++14 -Wall -g

OBJECTS = zeta.o

test: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ zeta.o

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $<

.PHONY: clean
clean:
	$(RM) $(OBJECTS)
	$(RM) -f test
