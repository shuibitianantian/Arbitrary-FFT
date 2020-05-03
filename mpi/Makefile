CXX = mpic++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native
LIBS = -lfftw3_mpi -lfftw3 -lm

TARGETS = $(basename $(wildcard *.cpp))

all: $(TARGETS)

%: %.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS)

.PHONY: all, clean
