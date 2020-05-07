// NaiveDTF_cuda is used to verify the correctness of other FFT version.

// struct NaiveDFT_cuda {
//     static constexpr char Name[] = "NaiveDFT_cuda";
//     const std::size_t N;

//     NaiveDFT_cuda(std::size_t N) : N(N) {}

//     ~NaiveDFT_cuda(){cudaDeviceReset();}

//     void dft(Comp* Y, const Comp* X){
//       using namespace std;

//       size_t xblock = std::sqrt(BLOCKSIZE); // currently support 2^n
//       dim3 dimBlock(xblock, xblock); // define the size of block
      
//       size_t xgrid = (N + dimBlock.x - 1) / dimBlock.x;
//       size_t ygrid = (N + dimBlock.y - 1) / dimBlock.y;
//       dim3 dimGrid(xgrid, ygrid); 

//       // initialize host and device of X
//       cuDoubleComplex* h_X = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));  // allocate memory to host
//       Comp_to_cuComp(X, h_X, N);  // convert Comp to cuda complex
//       cuDoubleComplex* d_X;
//       cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
//       cudaMemcpy(d_X, h_X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

//       // initialize host and device of Y
//       cuDoubleComplex* h_Y = (cuDoubleComplex*) malloc(N*ygrid*sizeof(cuDoubleComplex));
//       cuDoubleComplex* d_Y;
//       cudaMalloc(&d_Y, N*ygrid*sizeof(cuDoubleComplex));
//       for(int i = 0; i < N*ygrid; ++i)
//         h_Y[i] = make_cuDoubleComplex(0.0, 0.0);
//       cudaMemcpy(d_Y, h_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
//       cudaDeviceSynchronize();

//       double tt = omp_get_wtime();
//       dft_kernel2d<<<dimGrid, dimBlock>>>(d_X, d_Y, N, ygrid);
//       Check_CUDA_Error("Error");
//       reduction_2d<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, N, ygrid);
//       Check_CUDA_Error("Error");
//       cudaDeviceSynchronize();
//       cudaMemcpy(h_Y, d_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

//       for(int i = 0; i < N; ++i)
//         Y[i] = Comp(cuCreal(h_Y[i*ygrid]), cuCimag(h_Y[i*ygrid]));

//         // cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
//       free(h_Y);
//       free(h_X);
//     }

//     void idft(Comp* Y, const Comp* X) {
//       using namespace std;

//       size_t xblock = std::sqrt(BLOCKSIZE); // currently support 2^n
//       dim3 dimBlock(xblock, xblock); // define the size of block
      
//       size_t xgrid = (N + dimBlock.x - 1) / dimBlock.x;
//       size_t ygrid = (N + dimBlock.y - 1) / dimBlock.y;
//       dim3 dimGrid(xgrid, ygrid); 

//       // initialize host and device of X
//       cuDoubleComplex* h_X = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));  // allocate memory to host
//       Comp_to_cuComp(X, h_X, N);  // convert Comp to cuda complex
//       cuDoubleComplex* d_X;
//       cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
//       cudaMemcpy(d_X, h_X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

//       // initialize host and device of Y
//       cuDoubleComplex* h_Y = (cuDoubleComplex*) malloc(N*ygrid*sizeof(cuDoubleComplex));
//       cuDoubleComplex* d_Y;
//       cudaMalloc(&d_Y, N*ygrid*sizeof(cuDoubleComplex));
//       for(int i = 0; i < N*ygrid; ++i)
//         h_Y[i] = make_cuDoubleComplex(0.0, 0.0);
//       cudaMemcpy(d_Y, h_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
//       cudaDeviceSynchronize();

//       double tt = omp_get_wtime();
//       idft_kernel2d<<<dimGrid, dimBlock>>>(d_X, d_Y, N, ygrid);
//       Check_CUDA_Error("Error");
//       reduction_2d<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, N, ygrid);
//       Check_CUDA_Error("Error");
//       cudaDeviceSynchronize();
//       cudaMemcpy(h_Y, d_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  
//       for(int i = 0; i < N; ++i)
//         Y[i] = Comp(cuCreal(h_Y[i*ygrid]) / N, cuCimag(h_Y[i*ygrid]) / N);
//     cout << "[" << Name  << "] (idft) run time: " <<  omp_get_wtime() - tt << endl;

//       free(h_Y);
//       free(h_X);
//     }
// };