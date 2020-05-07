#include "fft_gpu.cu"

int main(){
    
    int N = 10000;

    Comp* input = (Comp*) malloc(N*sizeof(Comp));
    Comp* output = (Comp*) malloc(N*sizeof(Comp));

    for(int i = 0; i < N; ++i)
        input[i] = randComp(-10, 10);

    // run bluestein version of fft
    fft_cuda(1, 10000, input, output);
    
    free(input);
    free(output);
}