#include "fft_gpu.cu"

int main(int argc, char* argv[]){
    using namespace std;
    int N = atoi(argv[1]);
    
    Comp* input = (Comp*) malloc(N*sizeof(Comp));
    Comp* output = (Comp*) malloc(N*sizeof(Comp));
    Comp* original = (Comp*) malloc(N*sizeof(Comp));

    for(int i = 0; i < N; ++i)
        original[i] = input[i] = randComp(-10, 10);

    // run bluestein version of fft
    fft_cuda(1, N, input, output);
    
    // run benchmark, cufft
    benchmark(input, N);

    // report the error
    cout << "Error: " << error(output, input, N) << endl;

    free(input);
    free(output);
    free(original);

    return 0;
}