# computervision (UNDER DEVELOPMENT)

A collection of computer vision, data structures, and math utilites all implemented in C and accelerated with [CUDA v8.0].

Compile CUDA implementations with:
    ```nvcc -o EXE_NAME -lnppc -lnppi -lcublas -lcusolver FILENAME.cu```
    
# Design Notes
This library is designed with an Object Oriented architecture. This greatly simplifies the user-facing code and enables the math and image utilites to be platform-agnostic by allowing for abstract definitions and concrete implementations.

A minor annoyance of using this design in the C language is the need to pass a `self` argument to functions. (e.g. `imutil->convolve(imutil,image,kernel)`)

Furthermore: All functions assume they are operating on 32 bit floating point numbers (`float,Npp32f`) (if using CUDA, all Images are in device memory unless explicitly stated otherwise.)

# Core Components:

## [MatrixUtil]
The abstract definition of the MatrixUtil class, allowing concrete implementations for different platforms (i.e. CUDA, Metal)
### [PrimitiveMatrixUtil]
A plain-C implementation of the MatrixUtil.

### [CUDAMatrixUtil] 
(INCOMPLETE)
A [CUBLAS] and [CUSOLVER] implementation of the MatrixUtil.

### TODO: MetalMatrixUtil
For now, see my other repo [MetalUnity].

## [ImageUtil]

##### File Handling
(only PNGs are supported currently)
All file handling is currently done with [lodepng] 

##### NPP
Currently, the ImageUtil class is a concrete implementation, unlike the MatrixUtil class. It uses the [Nvidia Performance Primitives] library to perform common arithmetic, geometric and statistical operations on Images.

ToDo: Abstract this class and use MatrixUtil as low level image representations, not Npp32f.

# Example
```C
  int gw = 8; // Width of Gaussian Kernels
  int g1s = 5; // Sigma of Gaussian Kernel 1
  int g2s = 3; // Sigma of Gaussian Kernel 2
  int mw = 15; // Width of LocalMax window
  char* saves = "DoG.png"; // Filepath to save to
  char* path = "image.png"; // Filepath to load from
  //Grab ImageUtil
  ImageUtil* imutil = GetImageUtil(1); // 1 refers to CUDA Device Id
  //Load the image
  Image* im = imutil->loadImageFromFile(imutil,path);
  //Create the two gaussians
  Image* gauss1 = MakeGaussianKernel(imutil,gw,g1s); // defined in cv/Filters.c
  Image* gauss2 = MakeGaussianKernel(imutil,gw,g2s); // defined in cv/Filters.c
  //Get difference of gaussian kernel
  Image* DoGKernel = imutil->subtract(imutil,gauss1,gauss2);
  //Convolve
  Image* DoGImage = imutil->convolve(imutil,im,DoGKernel);
  //Find corners (local maximums)
  Image* corners = imutil->max(imutil,DoGImage,mw);
  //Save image
  imutil->saveImageToFile(imutil,corners,saves); // This is the only function that copies memory from device to host
```
![lena] ![corners]
[lodepng]: https://github.com/lvandeve/lodepng
[CUBLAS]:http://docs.nvidia.com/cuda/cublas/index.html#axzz4UOt5b3uc
[CUSOLVER]:http://docs.nvidia.com/cuda/cusolver/index.html#axzz4UOt5b3uc
[MetalUnity]:https://github.com/tylerpayne/MetalUnity
[Nvidia Performance Primitives]:http://docs.nvidia.com/cuda/npp/index.html#abstract
[CUDA v8.0]:https://developer.nvidia.com/cuda-toolkit

[MatrixUtil]:https://github.com/tylerpayne/computervision/blob/master/utils/MatrixUtil.h
[PrimitiveMatrixUtil]:https://github.com/tylerpayne/computervision/blob/master/utils/PrimitiveMatrixUtil.c
[CUDAMatrixUtil]:https://github.com/tylerpayne/computervision/blob/master/utils/CUDAMatrixUtil.cu
[ImageUtil]:https://github.com/tylerpayne/computervision/blob/master/utils/ImageUtil.h
[lena]:https://github.com/tylerpayne/computervision/blob/master/lena.png
[corners]:https://github.com/tylerpayne/computervision/blob/master/example.png
