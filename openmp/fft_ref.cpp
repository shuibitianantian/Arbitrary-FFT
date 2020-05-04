//
// Created by CONG YU on 5/3/20.
//
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <omp.h>
#include <fftw3.h>
#include "fft.cpp"

void genRandomComp(Comp* arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = randComp(-100, 100);
    }
}

void fftwToStdComp(fftw_complex* in, Comp* comp, int N) {
    for (int i = 0; i < N; i++) {
        comp[i] = {in[i][0], in[i][1]};
    }
}

void stdCompToFFTW(Comp* comp, fftw_complex* in, int N) {
    for (int i = 0; i < N; i++) {
        in[i][0] = comp[i].real();
        in[i][1] = comp[i].imag();
    }
}

int main(int argc, char** argv) {
    int N=1e7, threads, isDft;
    if (argc != 4) {
        printf("wrong");
        return 1;
    }
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &threads);
    sscanf(argv[3], "%d", &isDft); // 1 for dft 0 for idft

    double tt;
    Comp* compin = (Comp*) malloc(sizeof(Comp)*N);
    Comp* compout = (Comp*) malloc(sizeof(Comp)*N);
    genRandomComp(compin, N);

    // generate random data
    fftw_complex *in, *out;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    stdCompToFFTW(compin, in, N);
    Comp* fftwout = (Comp*) malloc(sizeof(Comp)*N);

    // run fftw
    fftw_plan p;
    if (isDft) {
        p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    } else {
        p = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    tt = omp_get_wtime();
    fftw_execute(p);
    std::cout << "FFTW Runtime: " << omp_get_wtime()-tt << std::endl;
    fftwToStdComp(out, fftwout, N);

    // run fft omp
    BluesteinFFT_omp bluesteinFftOmp(N);
    tt = omp_get_wtime();
    if (isDft) {
        bluesteinFftOmp.dft(compin, compout);
    } else {
        bluesteinFftOmp.idft(compin, compout);
    }
    double time = omp_get_wtime()-tt;
    std::cout << "OmpFFT Runtime: " << omp_get_wtime()-tt << std::endl;

    // compare error
    double err = error(compout, fftwout, N);
    std::cout << "Err " << err << std::endl;

    // free
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    free(compin);
    free(compout);
    return 0;
}

