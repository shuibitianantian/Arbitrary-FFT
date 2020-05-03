CXX = mpic++
CXXFLAGS += -std=c++17 -O3 -march=native -mtune=native

TARGETS = $(basename $(wildcard *.cpp))

all: $(TARGETS)

fftw-mpi: LIBS += -lfftw3_mpi -lfftw3

%: %.cpp *.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS)

.PHONY: all, clean
