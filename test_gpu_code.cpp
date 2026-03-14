#include<bits/stdc++.h> 

#include "varsize_grouped_batched_inverse_lib.cu"
#include<omp.h> 

int main()
{
    const int K = 5000000;                 // number of matrices
  const int nMin = 25, nMax = 125;     // variable sizes
  const double vramSafety = 0.80;     // use 80% of reported free VRAM

  std::vector<Eigen::MatrixXcd> A(K), Ainv(K);
  std::vector<int> nOf(K);

  #pragma omp parallel
  {
    std::mt19937_64 rng(1234ULL + 1337ULL*(unsigned)omp_get_thread_num());
    std::uniform_int_distribution<int> ndist(nMin, nMax);
    std::uniform_real_distribution<double> rdist(-1.0, 1.0);

    #pragma omp for schedule(static)
    for (int k = 0; k < K; ++k) {
      int n = ndist(rng);
      nOf[k] = n;

      A[k].resize(n,n);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          A[k](i,j) = {rdist(rng), rdist(rng)};

      // diagonal shift for better conditioning
      A[k].diagonal().array() += std::complex<double>(double(n), 0.0);
    }
  }

  
  //invert_varsize_mats_batched_gpu(A, Ainv, 0.80, 128);

  /*
  for(int i=0; i<Ainv.size(); ++i)
  {
    std::cout << "mat: " << i << std::endl; 
    std::cout << Ainv[i] << std::endl; 
  }
  */

}