/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <cstdio>
#define THREADS_PB 1024
#define BLOCKS(n) ((n + THREADS_PB -1) / THREADS_PB)
__global__
void baselineHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numVals) return;
    atomicAdd(&histo[vals[tid]], 1);
}
__global__
void localHisto(const unsigned int* const vals,
               unsigned int* const histo,   
               int numBins,
               int numVals)
{
    extern __shared__ unsigned int lh[];
    for (int i=0; i<numBins; i+=blockDim.x){
        lh[i + threadIdx.x] = 0;
    }
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numVals) return;
    atomicAdd(&lh[vals[tid]], 1);
    __syncthreads(); 
    for (int i=0; i<numBins; i+=blockDim.x){
        atomicAdd(&histo[i + threadIdx.x], lh[i + threadIdx.x]);
    }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
    localHisto<<<BLOCKS(numElems), THREADS_PB, numBins * sizeof(unsigned int)>>>(d_vals, d_histo, numBins, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
