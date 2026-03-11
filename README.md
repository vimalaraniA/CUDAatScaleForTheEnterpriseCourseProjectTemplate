# GPU Accelerated Image Processing and Parallel Merge Sort using CUDA

## Overview

This project demonstrates **GPU-accelerated image processing using NVIDIA CUDA**.

The program processes a dataset of images using **parallel GPU computation** with custom CUDA kernels. The GPU processes thousands of pixels simultaneously, allowing faster execution compared to traditional CPU processing.

The project performs the following operations:

- **Image loading** from a dataset folder
- **GPU blur filtering** using CUDA kernels
- **Saving processed images**
- **GPU merge sort** on pixel data

This project highlights the **power of parallel computing using CUDA GPUs** for large-scale image processing tasks.

---

# Project Structure

```
ImageProcessingGPU/
│
├── bin/                # Compiled CUDA executables
├── data/               # Input images
├── output/             # Processed output images
│
├── src/                # Source code
│   ├── main.cu         # Main CUDA program
│   ├── merge.cu        # GPU merge sort implementation
│   └── utils.h         # Helper functions
│
├── execution_results/  # Proof of execution
│   ├── before_image.png
│   ├── after_blur.png
│   └── execution_log.txt
│
├── README.md
├── INSTALL
├── Makefile
└── run.sh
```

---

# System Requirements

Ensure your system contains:

- **NVIDIA GPU**
- **CUDA Toolkit installed**
- **CUDA compatible drivers**
- **C++ compiler supporting C++17**

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ImageProcessingGPU.git
cd ImageProcessingGPU
```

Build the project using **Makefile**:

```bash
make all
```

This will generate the executable inside the `bin/` directory.

---

# Running the Program

The program accepts **command line arguments**.

Run the executable:

```bash
./bin/image_processing data/ output/
```

### Arguments

| Argument | Description |
|--------|-------------|
| `data/` | Folder containing input images |
| `output/` | Folder where processed images will be saved |

---

# How the Program Works

## Image Processing using CUDA

The program performs the following steps:

1. Loads images from the `data/` directory.
2. Transfers image data from **CPU memory to GPU memory**.
3. Launches a **CUDA blur kernel** where each thread processes a pixel.
4. Copies the processed image back to CPU memory.
5. Saves the processed image in the `output/` directory.

CUDA configuration used:

```
Threads per block: 16 x 16
Grid size: Computed dynamically
```

This configuration allows thousands of GPU threads to execute simultaneously.

---

# GPU Merge Sort

The project also demonstrates **parallel merge sort using CUDA**.

Steps involved:

1. Pixel values are converted into a **1-D array**.
2. The array is transferred to **GPU memory**.
3. CUDA kernels perform **parallel merge operations**.
4. The sorted array is copied back to the CPU.

This demonstrates GPU efficiency for **large-scale data computations**.

---

# CUDA Kernel Example

```cpp
__global__ void blurKernel(unsigned char* input,
                           unsigned char* output,
                           int width,
                           int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int sum = 0;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                sum += input[(y + i) * width + (x + j)];
            }
        }

        output[y * width + x] = sum / 9;
    }
}
```

Each GPU thread processes **one pixel**, allowing massive parallel computation.

---

# Example Output

Program execution:

```
Processing images from: data/

Processed image: data/image1.png
Processed image: data/image2.png
Processed image: data/image3.png

All images processed successfully.
```

Processed images are saved inside the `output/` folder.

---

# Execution Artifacts

Proof of execution is stored in the `execution_results/` folder.

Files included:

```
before_image.png
after_blur.png
execution_log.txt
```

Example log output:

```
Input Image Resolution: 2048 x 2048

CPU Execution Time: 1.25 seconds
GPU Execution Time: 0.19 seconds

CUDA blur kernel executed successfully.
GPU merge sort completed.
```

---

# Challenges Faced

During development the following challenges were encountered:

- Managing **GPU memory allocation**
- Synchronizing **host and device operations**
- Configuring optimal **thread block and grid dimensions**

These challenges were solved through careful CUDA kernel configuration and memory management.

---

# Results

The project demonstrates that **GPU execution significantly outperforms CPU execution** for large-scale image processing tasks.

Using CUDA kernels allows thousands of pixels to be processed simultaneously.

---

# Future Improvements

Possible future enhancements include:

- **Real-time video processing using CUDA**
- Implementing additional filters such as **edge detection**
- Integration with **deep learning models**
- Optimized **parallel sorting algorithms**

---

# Conclusion

This project demonstrates the advantages of **GPU-based parallel computing using CUDA**.

Image processing operations that normally require significant CPU time can be executed much faster using GPU kernels.

---

# License

This project is developed for **educational purposes** as part of the **GPU Specialization Capstone Project**.
