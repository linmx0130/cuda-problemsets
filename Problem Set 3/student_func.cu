/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <cstdio>

__global__ void reduce_find_min_value(const size_t N, float *buf){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (int sl = (N+1) / 2; sl>1; sl = (sl + 1)>>1){
        if (tid < sl){
            buf[tid] = min(buf[tid], buf[tid + sl]);
        }
    }
    buf[0] = min(buf[0], buf[1]);
}

__global__ void reduce_find_max_value(const size_t N, float *buf){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (int sl = (N+1) / 2; sl>1; sl = (sl + 1)>>1){
        if (tid < sl){
            buf[tid] = max(buf[tid], buf[tid + sl]);
        }
    }
    buf[0] = max(buf[0], buf[1]);
}

__global__ void getBinOfInput(const float* const input, float lumMin, float lumRange, size_t numBins, size_t dataCount, unsigned int* col_o){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bin_num = 0;
    if (tid < dataCount){
        bin_num = min((size_t)((input[tid] - lumMin) / lumRange * numBins), numBins-1);
        atomicAdd(&col_o[bin_num], 1);
    }
}
__global__ void getCdf(unsigned int *d_bin, const int numBins, unsigned int* const d_cdf){
    for (int i=1;i<numBins; ++i){
        d_cdf[i] = d_bin[i-1] + d_cdf[i-1];
    }
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    float *d_buf;
    int N = numRows * numCols;
    if (N & 1) N += 1;
    checkCudaErrors(cudaMalloc(&d_buf, sizeof(float) * N));

    cudaMemset(d_buf, 1.0f, sizeof(float) * N);
    cudaMemcpy(d_buf, d_logLuminance, numRows*numCols, cudaMemcpyDeviceToDevice);
    reduce_find_min_value<<< (N + 1023)/1024, 1024>>>(N, d_buf);
    cudaMemcpy(&min_logLum, d_buf, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(d_buf, 0.0f, sizeof(float) * N);
    cudaMemcpy(d_buf, d_logLuminance, numRows*numCols, cudaMemcpyDeviceToDevice);
    reduce_find_max_value<<< (N + 1023)/1024, 1024>>>(N, d_buf);
    cudaMemcpy(&max_logLum, d_buf, sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaFree(d_buf));

    float lumRange = max_logLum - min_logLum;
    N = numRows * numCols;
    unsigned int *d_bin;
    checkCudaErrors(cudaMalloc(&d_bin, sizeof(unsigned int) * numBins));
    cudaMemset(d_bin, 0 ,sizeof(unsigned int) * numBins);
    getBinOfInput<<<(N+1023)/1024, 1024>>>(d_logLuminance, min_logLum, lumRange, numBins, N, d_bin);
    cudaMemset(d_cdf, 0 ,sizeof(numBins));
    getCdf<<<1, 1>>>(d_bin, numBins, d_cdf);
}
