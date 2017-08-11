//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#define NUM_THREAD 512
#define NUM_BLOCK(n) ((n + NUM_THREAD - 1)/NUM_THREAD)

#include "utils.h"
#include <thrust/host_vector.h>
#include <cstdio>
__global__ void split_channels(const uchar4* const d_sourceImg, 
                          float * const d_R, 
                          float * const d_G,
                          float * const d_B, size_t total_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_size) return;
    d_R[tid] = d_sourceImg[tid].x;
    d_G[tid] = d_sourceImg[tid].y;
    d_B[tid] = d_sourceImg[tid].z;
}
__global__ void get_mask(const uchar4* const simg, char * mask, size_t N){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid>=N) return;
  uchar4 p = simg[tid];
  char f = (p.x == 255) && (p.y == 255) && (p.z == 255);
  mask[tid] = !f;
}

__global__ void get_border_pixel(const char *mask, size_t numRowsSource, size_t numColsSource, char* border, char* interior){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numRowsSource * numColsSource) return;
  char f = 1;
  int x = tid / numColsSource, y = tid % numColsSource;
  if (x-1 >=0) 
      f = f && mask[ (x-1) * numColsSource + y];
  if (x+1 < numRowsSource) f = f && mask[ (x+1) * numColsSource + y];
  if (y-1 >=0) f = f && mask[ x * numColsSource + y - 1];
  if (y+1 < numColsSource) f = f && mask[ x * numColsSource + y + 1];
  border[tid] = mask[tid] && (!f);
  interior[tid] = mask[tid] && f;
}

__global__ void set_interior(float *dest, const char *interior, const float* const src, size_t total_size){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= total_size) return;
  if (interior[tid]) dest[tid] = src[tid];
}

__global__ void calc_iteration(const float *d_buf1, const float* target, const char* mask, const char* interior, const float* dsource, 
                               size_t numRowsSource, size_t numColsSource, float* d_buf2){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numRowsSource*numColsSource) return;
  int x = tid / numColsSource;
  int y = tid % numColsSource;
  int pid;
  if (interior[tid]){
      float A=0.f, B=0.f, D=0.f;
      if (x-1>=0){
        pid = tid - numColsSource;
        A += d_buf1[pid] * interior[pid]; 
        B += target[pid] * (1-interior[pid]); 
        D += 1;
      }
      if (x + 1 < numRowsSource){
        pid = tid + numColsSource;
        A += d_buf1[pid] * interior[pid]; 
        B += target[pid] * (1-interior[pid]); 
        D += 1;
      }
      if (y-1 >= 0){
        pid = tid -1;
        A += d_buf1[pid] * interior[pid]; 
        B += target[pid] * (1 - interior[pid]); 
        D += 1;
      }
      if (y+1 < numColsSource){
        pid = tid + 1;
        A += d_buf1[pid] * interior[pid]; 
        B += target[pid] * (1-interior[pid]); 
        D += 1;
      }

      float tmp = (A + B + dsource[tid]) / D;
      tmp = (tmp >= 255.f) ? 255: tmp;
      tmp = (tmp <0.f) ? 0.f : tmp;
      d_buf2[tid] = tmp;
  }
  //else{
  //  d_buf2[tid] = d_buf1[tid];
 // }
}

__global__ void get_sdiff(const float* const src, float* odiff, size_t numRowsSource, size_t numColsSource){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numRowsSource * numColsSource) return;
  int x = tid / numColsSource;
  int y = tid % numColsSource;
  float sum = 0.f;
  if (x-1>=0){
    sum += src[tid] - src[tid - numColsSource];
  }
  if (x + 1 < numRowsSource){
    sum += src[tid] - src[tid + numColsSource];
  }
  if (y-1 >= 0){
    sum += src[tid] - src[tid - 1];
  }
  if (y+1 < numColsSource){
    sum += src[tid] - src[tid + 1];
  }
  odiff[tid] = sum;
}
__global__ void combind_result(const float* const red, const float* const green, const float* const blue, 
                          size_t total_size, uchar4* const d_blend){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= total_size) return;
  uchar4 output_pixel = make_uchar4((char)red[tid], (char)green[tid], (char)blue[tid], 255);
  d_blend[tid] = output_pixel;
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  float *d_channels[3];
  float *d_dest[3];
  uchar4 *d_sourceImg, *d_destImg;
  size_t total_size = numRowsSource * numColsSource;
  //alloc resources
  cudaMalloc(&d_sourceImg, sizeof(uchar4) * total_size);
  cudaMalloc(&d_destImg, sizeof(uchar4) * total_size);
  for (int i=0;i<3;++i){
    cudaMalloc(&d_channels[i], sizeof(float) * total_size);
    cudaMalloc(&d_dest[i], sizeof(float) * total_size);
  }
  cudaMemcpyAsync(d_sourceImg, h_sourceImg, sizeof(uchar4) * total_size, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(d_destImg, h_destImg, sizeof(uchar4) * total_size, cudaMemcpyHostToDevice, s2);
  char *d_mask, *d_border, *d_interior;
  cudaMalloc(&d_mask, sizeof(char) * total_size);
  cudaMalloc(&d_border, sizeof(char) * total_size);
  cudaMalloc(&d_interior, sizeof(char) * total_size);
  
  //prepare input data
  split_channels<<<NUM_BLOCK(total_size), NUM_THREAD, 0, s1>>> (d_sourceImg, d_channels[0], d_channels[1], d_channels[2], total_size);
  split_channels<<<NUM_BLOCK(total_size), NUM_THREAD, 0, s2>>> (d_destImg, d_dest[0], d_dest[1], d_dest[2], total_size);
 
  //get mask and border 
  get_mask<<<NUM_BLOCK(total_size), NUM_THREAD, 0, s1>>> (d_sourceImg, d_mask, total_size);
  get_border_pixel<<<NUM_BLOCK(total_size), NUM_THREAD, 0, s1>>> (d_mask, numRowsSource, numColsSource, d_border, d_interior);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  float *d_buf1, *d_buf2, *d_sdiff;
  cudaMalloc(&d_buf1, sizeof(float) * total_size);
  cudaMalloc(&d_buf2, sizeof(float) * total_size);
  cudaMalloc(&d_sdiff, sizeof(float) * total_size);
  for (int color = 0; color<3; ++color){
    cudaMemcpy(d_buf1, d_dest[color], sizeof(float)*total_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_buf2, d_dest[color], sizeof(float)*total_size, cudaMemcpyDeviceToDevice);
    set_interior<<<NUM_BLOCK(total_size), NUM_THREAD>>> (d_buf1, d_interior, d_channels[color], total_size);
    get_sdiff<<<NUM_BLOCK(total_size), NUM_THREAD>>> (d_channels[color], d_sdiff, numRowsSource, numColsSource);

    for (int itr =0; itr < 800; ++itr){
      calc_iteration<<<NUM_BLOCK(total_size), NUM_THREAD>>> (d_buf1, d_dest[color], d_mask, d_interior, d_sdiff, numRowsSource, numColsSource, d_buf2);
      float* k = d_buf1; d_buf1 = d_buf2; d_buf2=k;
    }
    cudaMemcpy(d_dest[color], d_buf1, sizeof(float)*total_size, cudaMemcpyDeviceToDevice);
  }
  uchar4 *d_blend;
  cudaMalloc(&d_blend, sizeof(uchar4) * total_size); 
  combind_result<<<NUM_BLOCK(total_size), NUM_THREAD>>> (d_dest[0], d_dest[1], d_dest[2], total_size, d_blend);
  //combind_result<<<NUM_BLOCK(total_size), NUM_THREAD>>> (d_channels[0], d_channels[1], d_channels[2], total_size, d_blend);
  cudaMemcpy(h_blendedImg, d_blend, sizeof(uchar4)*total_size, cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //clean up
  cudaFree(d_blend);
  cudaFree(d_buf1);
  cudaFree(d_buf2);
  cudaFree(d_border);
  cudaFree(d_interior);
  cudaFree(d_mask);
  cudaFree(d_sourceImg);
  for (int i=0;i<3;++i) {
    cudaFree(d_channels[i]);
    cudaFree(d_dest[i]);
  }
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
}
