MARCH ?= -march=native -mtune=native

CXXFLAGS += -std=c++17 -O3 $(MARCH)

TARGETS = $(basename $(wildcard *.cpp))

all: $(TARGETS)

fftw-mpi: LIBS += -lfftw3_mpi -lfftw3

fftw-mpi: CXX = mpic++
mpi: 			CXX = mpic++
mpi-h: 		CXX = mpic++
mpi-nb: 	CXX = mpic++
mpi-prog: CXX = mpic++
testin:		CXX = mpic++

%: %.cpp *.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS)

.PHONY: all, clean
