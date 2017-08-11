//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <cstdio>

#define THREAD_COUNT 512
#define BLOCKS(n) ((n + THREAD_COUNT - 1) / THREAD_COUNT)
/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void get_digit_at_k(unsigned int * const d_input, const size_t n, unsigned int k, unsigned int* const d_buf){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    d_buf[tid] = (d_input[tid] & (1<<k)) >> k;
}

__global__ void reduce_sum_kernel(unsigned int * const d_input, const size_t n, const size_t step_start, unsigned int * const d_output){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >=n) return;
    if (tid < step_start) {
        d_output[tid] = d_input[tid];
    }else{
        d_output[tid] = d_input[tid] + d_input[tid - step_start];
    }
}

void get_acc_sum(unsigned int * const d_input, const size_t n, unsigned int * const d_output){
    unsigned int *d_buf1, *d_buf2;
    cudaMalloc(&d_buf1, sizeof(unsigned int) * n);
    cudaMalloc(&d_buf2, sizeof(unsigned int) * n);
    cudaMemset(d_buf2, 0 ,sizeof(unsigned int) * n);
    cudaMemcpy(d_buf1, d_input, sizeof(unsigned int) * n, cudaMemcpyDeviceToDevice);

    for (unsigned int step_start = 1; step_start<n; step_start<<=1){
        reduce_sum_kernel<<<BLOCKS(n), THREAD_COUNT>>>(d_buf1, n, step_start, d_buf2);
        unsigned int *k = d_buf1;
        d_buf1 = d_buf2;
        d_buf2 = k;
    }

    cudaMemcpy(d_output, d_buf1, sizeof(unsigned int) * n, cudaMemcpyDeviceToDevice);
    cudaFree(d_buf1);
    cudaFree(d_buf2);
}

__global__ void not_op_on_array(unsigned int * const d_input, const size_t n){
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= n) return;
    d_input[t] = !d_input[t];
}
__global__ void put_data(unsigned int *d_input_vals, unsigned int *d_input_pos, size_t n, 
                         unsigned int *acc_sum_0, unsigned int *acc_sum_1, size_t zero_count,
                         unsigned int *is_zero, unsigned int *d_output_vals, unsigned int *d_output_pos){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid>=n) return;
    size_t add_base = (is_zero[tid] ^ 1) * zero_count;
    unsigned int * relative_add = is_zero[tid] ? acc_sum_0 : acc_sum_1;
    
    size_t targetIdx = add_base + relative_add[tid] - 1;
    d_output_vals[targetIdx] = d_input_vals[tid];
    d_output_pos[targetIdx] = d_input_pos[tid];
}
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    int digit_k = 0;
    unsigned int one_count, zero_count;
    unsigned int *d_buf1, *d_acc_sum_1, *d_acc_sum_0;
    unsigned int *d_val_buf;
    unsigned int *d_pos_buf;
    cudaMalloc(&d_buf1, sizeof(unsigned int) * numElems);
    cudaMalloc(&d_acc_sum_1, sizeof(unsigned int) * numElems);
    cudaMalloc(&d_acc_sum_0, sizeof(unsigned int) * numElems);
    cudaMalloc(&d_val_buf, sizeof(unsigned int) * numElems);
    cudaMalloc(&d_pos_buf, sizeof(unsigned int) * numElems);
    cudaMemcpy(d_val_buf, d_inputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice); 
    cudaMemcpy(d_pos_buf, d_inputPos, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice); 

    while (true){
        get_digit_at_k<<<BLOCKS(numElems), THREAD_COUNT>>>(d_val_buf, numElems, digit_k, d_buf1);
        get_acc_sum(d_buf1, numElems, d_acc_sum_1); 
        //check if there is 1 on dight k
        cudaMemcpy(&one_count, d_acc_sum_1 + numElems - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (one_count == 0){
            break;
        }
        zero_count = numElems - one_count;
        not_op_on_array<<<BLOCKS(numElems), THREAD_COUNT>>>(d_buf1, numElems);
        get_acc_sum(d_buf1, numElems, d_acc_sum_0); 
        put_data<<<BLOCKS(numElems), THREAD_COUNT>>>(d_val_buf, d_pos_buf, numElems, d_acc_sum_0, d_acc_sum_1, zero_count, d_buf1, d_outputVals, d_outputPos);
        cudaMemcpy(d_val_buf, d_outputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_pos_buf, d_outputPos, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice);
        digit_k = digit_k + 1;
    }
    cudaFree(d_buf1);
    cudaFree(d_acc_sum_1);
    cudaFree(d_acc_sum_0);
    cudaFree(d_val_buf);
    cudaFree(d_pos_buf);
}
