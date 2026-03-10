# GPU Image Processing Project

## Overview

This project demonstrates **GPU-accelerated image processing using NVIDIA CUDA**.

The program processes a dataset of images using **parallel GPU computation** with custom CUDA kernels. The GPU processes thousands of pixels simultaneously, allowing faster execution compared to traditional CPU processing.

The project performs the following operations:

- Image loading from a dataset
- GPU blur filtering using CUDA kernels
- Saving processed images
- GPU merge sort on pixel data to demonstrate additional GPU computation

This project highlights the **power of parallel computing using CUDA GPUs** for large-scale image processing tasks.

---

# Code Organization

```
ImageProcessingGPU/
│
├── bin/                # Compiled CUDA executables
├── data/               # Input images
├── lib/                # External libraries (if required)
├── src/                # Source code
│   ├── main.cu         # Main CUDA program
│   ├── merge.cu        # GPU merge sort implementation
│   └── utils.h         # Helper functions
│
├── output/             # Processed output images
│
├── README.md           # Project documentation
├── INSTALL             # Installation instructions
├── Makefile
└── run.sh
```

---

# Installation

## Linux / MacOS

Ensure that your system contains:

- **NVIDIA GPU**
- **CUDA Toolkit installed**
- **CUDA compatible drivers**

Clone the repository:

```bash
git clone https://github.com/yourusername/ImageProcessingGPU.git
cd ImageProcessingGPU
```

Build the project:

```bash
make all
```

Or using CMake:

```bash
mkdir build
cd build
cmake ..
make
```

Run the executable:

```bash
./bin/image_processing
```

---

## Windows (Visual Studio)

1. Install **Visual Studio with CUDA Toolkit**
2. Open the project solution file
3. Build the project
4. Run the executable from the **bin/** directory

---

# Usage

1. Place input images inside the **data/** directory.

2. Run the program:

```bash
./bin/image_processing
```

3. The program will:

- Load images
- Process them on GPU
- Save results to **output/** folder

4. Parameters such as **thread size, grid size, and filters** can be modified in `main.cu`.

---

# main.cu

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "merge.cu"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

__global__ void blurKernel(unsigned char* input,unsigned char* output,int width,int height){

    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;

    if(x>=1 && x<width-1 && y>=1 && y<height-1){

        int sum=0;

        for(int i=-1;i<=1;i++)
            for(int j=-1;j<=1;j++)
                sum+=input[(y+i)*width+(x+j)];

        output[y*width+x]=sum/9;
    }
}

int main(){

    std::string data_folder="data/";
    std::string output_folder="output/";

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid;

    for(const auto& entry:fs::directory_iterator(data_folder)){

        std::string img_path=entry.path().string();

        int width,height,channels;

        unsigned char* img=stbi_load(img_path.c_str(),&width,&height,&channels,1);

        if(!img){
            std::cout<<"Image load failed"<<std::endl;
            continue;
        }

        unsigned char *d_input,*d_output;

        cudaMalloc(&d_input,width*height*sizeof(unsigned char));
        cudaMalloc(&d_output,width*height*sizeof(unsigned char));

        cudaMemcpy(d_input,img,width*height*sizeof(unsigned char),cudaMemcpyHostToDevice);

        blocksPerGrid.x=(width+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.y=(height+threadsPerBlock.y-1)/threadsPerBlock.y;

        blurKernel<<<blocksPerGrid,threadsPerBlock>>>(d_input,d_output,width,height);

        cudaDeviceSynchronize();

        unsigned char* output_img=new unsigned char[width*height];

        cudaMemcpy(output_img,d_output,width*height*sizeof(unsigned char),cudaMemcpyDeviceToHost);

        std::string output_path=output_folder+entry.path().filename().string();

        stbi_write_png(output_path.c_str(),width,height,1,output_img,width);

        std::cout<<"Processed image: "<<img_path<<std::endl;

        delete[] output_img;
        stbi_image_free(img);
        cudaFree(d_input);
        cudaFree(d_output);
    }

    return 0;
}
```

---

# merge.cu

```cpp
#include <cuda_runtime.h>
#include <algorithm>

__device__ void gpu_bottomUpMerge(long* source,long* dest,long start,long middle,long end){

    long i=start;
    long j=middle;

    for(long k=start;k<end;k++){

        if(i<middle && (j>=end || source[i]<=source[j])){
            dest[k]=source[i++];
        }
        else{
            dest[k]=source[j++];
        }
    }
}

__global__ void gpu_mergesort(long* source,long* dest,long size,long width){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    long start=idx*width;

    if(start>=size) return;

    long middle=min(start+(width/2),size);
    long end=min(start+width,size);

    gpu_bottomUpMerge(source,dest,start,middle,end);
}
```

---

# How It Works

### main.cu

- Loads images from the **data/** directory
- Transfers image data to **GPU memory**
- Executes a **CUDA blur kernel**
- Saves processed images to the **output/** folder

### merge.cu

- Implements **parallel merge sort using CUDA**
- Demonstrates GPU processing for **large array computations**

---

# Features

- **GPU accelerated image processing**
- **CUDA kernel implementation**
- **Parallel blur filtering**
- **GPU merge sort example**
- **Batch image processing**
- **Automatic output generation**

---

# Results

The program successfully processes multiple images using **CUDA GPU kernels**.

Example output:

```
Processed image: data/image1.png
Processed image: data/image2.png
Processed image: data/image3.png
```

Processed images are saved inside the **output/** folder.

---

# Conclusion

This project demonstrates the advantages of **GPU-based parallel computing** using CUDA. Image processing tasks that would normally take significant CPU time can be executed much faster using GPU kernels.

Future improvements may include:

- **Real-time video processing**
- **GPU based object detection**
- **Deep learning integration**
