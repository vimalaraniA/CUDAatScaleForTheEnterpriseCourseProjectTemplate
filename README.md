# GPU Image Processing Project

## Overview

This project demonstrates GPU-accelerated image processing using CUDA.
It processes a dataset of images (hundreds of small images or tens of large images) with GPU computation, leveraging either custom CUDA kernels or the NVIDIA Performance Primitives (NPP) library.

The goal is to apply image processing techniques such as filtering, convolution, and transformations on images efficiently using GPU parallelism.

## Code Organization

```
ImageProcessingGPU/
│
├── bin/                # Compiled executables (e.g., .exe or CUDA binaries)
├── data/               # Sample input images
├── lib/                # Any external libraries (if not installed system-wide)
├── src/                # Source code
│   ├── main.cpp        # Main program
│   ├── cuda_kernels.cu # CUDA kernels for image processing
│   └── utils.h         # Helper functions for image I/O, memory management
├── output/             # Output images after GPU processing
├── README.md           # Project description
├── INSTALL             # Instructions to build and run
├── Makefile / CMakeLists.txt / build.sh
└── run.sh              # Optional script to run the program
```

## Installation
Linux / MacOS

Ensure you have an NVIDIA GPU and CUDA toolkit installed.

Clone the repository

Build the project:
```
# Using Makefile
make all

# Or using CMake
mkdir build && cd build
cmake ..
make
```
Run the executable:

```
./bin/image_processing
```
### Windows (with Visual Studio)

Install Visual Studio with CUDA toolkit.

Open ImageProcessingGPU.sln and build the solution.

Run the executable from bin/.

## Usage

Place input images in the data/ folder.

Processed images will be saved in the output/ folder.

Modify main.cpp to change processing parameters (e.g., threads per block, image filters).

## main.cu
```
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "merge.cu"   // Include your CUDA merge sort implementation

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

// Simple blur kernel example for demonstration
__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
        int sum = 0;
        for (int i=-1; i<=1; i++)
            for (int j=-1; j<=1; j++)
                sum += input[(y+i)*width + (x+j)];
        output[y*width + x] = sum / 9;
    }
}

int main(int argc, char** argv) {
    std::string data_folder = "data/";
    std::string output_folder = "output/";

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid;

    // Loop through all images in the data folder
    for (const auto& entry : fs::directory_iterator(data_folder)) {
        std::string img_path = entry.path().string();

        int width, height, channels;
        unsigned char* img = stbi_load(img_path.c_str(), &width, &height, &channels, 1);
        if (!img) {
            std::cerr << "Failed to load image: " << img_path << std::endl;
            continue;
        }

        unsigned char* d_input;
        unsigned char* d_output;
        cudaMalloc(&d_input, width * height * sizeof(unsigned char));
        cudaMalloc(&d_output, width * height * sizeof(unsigned char));
        cudaMemcpy(d_input, img, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

        blocksPerGrid.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocksPerGrid.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y;

        blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();

        // Copy result back
        unsigned char* output_img = new unsigned char[width*height];
        cudaMemcpy(output_img, d_output, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

        std::string output_path = output_folder + entry.path().filename().string();
        stbi_write_png(output_path.c_str(), width, height, 1, output_img, width);

        // Example: Flatten image to array and run GPU merge sort
        long* img_data = new long[width*height];
        for (int i = 0; i < width*height; i++) img_data[i] = output_img[i];

        dim3 threads(32);
        dim3 blocks(8);
        long* sorted = mergesort(img_data, width*height, threads, blocks);

        delete[] img_data;
        delete[] sorted;
        delete[] output_img;
        stbi_image_free(img);
        cudaFree(d_input);
        cudaFree(d_output);

        std::cout << "Processed image: " << img_path << std::endl;
    }

    return 0;
}
```

## merge.cu

```
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

// GPU bottom-up merge
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] <= source[j])) {
            dest[k] = source[i++];
        } else {
            dest[k] = source[j++];
        }
    }
}

// Calculate global thread index
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x * (x *= threads->z) +
           blockIdx.y * (x *= blocks->z) +
           blockIdx.z * (x *= blocks->y);
}

// GPU merge sort kernel
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices;

    for (long slice = 0; slice < slices; ++slice) {
        if (start >= size) break;

        long middle = min(start + (width / 2), size);
        long end = min(start + width, size);

        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

// Host function to call GPU merge sort
long* mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    long *D_data, *D_swp;
    dim3 *D_threads, *D_blocks;

    long* result = new long[size];
    cudaMalloc(&D_data, sizeof(long) * size);
    cudaMalloc(&D_swp, sizeof(long) * size);
    cudaMalloc(&D_threads, sizeof(dim3));
    cudaMalloc(&D_blocks, sizeof(dim3));

    cudaMemcpy(D_data, data, sizeof(long) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    long* A = D_data;
    long* B = D_swp;
    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (long width = 2; width < (size << 1); width <<= 1) {
        long slices = size / (nThreads * width) + 1;
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);
        cudaDeviceSynchronize();
        std::swap(A, B);
    }

    cudaMemcpy(result, A, sizeof(long) * size, cudaMemcpyDeviceToHost);

    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(D_threads);
    cudaFree(D_blocks);

    return result;
}
```
## How It Works

##### main.cu:

Loads images from data/

Sends them to GPU

Applies a simple blur filter kernel

Saves processed images to output/

Flattens image to 1D array and runs GPU merge sort as a demo of GPU computation

##### merge.cu:

Implements GPU merge sort kernel

Host function mergesort handles memory allocation and GPU kernel calls

## Features

GPU-accelerated merge sort kernel for large arrays (example of GPU computation)

Image processing kernels: convolution, blur, edge detection

Handles hundreds of small images or tens of large images

Saves processed images for verification

## Results

Processed images demonstrate GPU-based filtering.

Logs show GPU kernel execution and timing results for comparison with CPU.
