# computervision (UNDER DEVELOPMENT)

A collection of computer vision, data structures, and math utilities all implemented in C and accelerated with [CUDA v8.0].

# Design Notes
This library is designed with an Object Oriented architecture. This greatly simplifies the user-facing code and enables the math and image utilities to be platform-agnostic by allowing for abstract definitions and concrete implementations.

A minor annoyance of using this design in the C language is the need to pass a `self` argument to functions. \n(e.g. `imutil->convolve(imutil,image,kernel)`)

Furthermore: All functions assume they are operating on 32 bit floating point numbers (`float, Npp32f`)

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
The abstract definition of the ImageUtil class, allowing concrete implementations for different platforms (i.e. CUDA, Metal). ImageUtil depends on MatrixUtil's Matrix initializers for the low level representations of images.

##### File Handling
(only PNGs are supported currently)
All file handling is currently done with [lodepng]

### [CUDAImageUtil]

##### NPP
 CUDAImageUtil uses the [Nvidia Performance Primitives] library to perform common arithmetic, geometric and statistical operations on Images.



# Example
```C
  int gw = 8; // Width of Gaussian Kernels
  int g1s = 5; // Sigma of Gaussian Kernel 1
  int g2s = 3; // Sigma of Gaussian Kernel 2
  int mw = 15; // Width of LocalMax window
  char* saves = "DoG.png"; // Filepath to save to
  char* path = "image.png"; // Filepath to load from
  MatrixUtil* matutil = GetCUDAMatrixUtil(1); // Get CUDA MatrixUtil
  ImageUtil* imutil = GetCUDAImageUtil(matutil); // Get CUDA Image Util
  //Load the image
  Image* in = imutil->loadImageFromFile(imutil,path);
  Image* im = imutil->resample(imutil,in,256,256);
  //Create the two gaussians
  Image* gauss1 = MakeGaussianKernel(imutil,gw,g1s);
  Image* gauss2 = MakeGaussianKernel(imutil,gw,g2s);
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
[CUDAImageUtil]:https://github.com/tylerpayne/computervision/blob/master/utils/CUDAImageUtil.h
[lena]:https://github.com/tylerpayne/computervision/blob/master/lena256.png
[corners]:https://github.com/tylerpayne/computervision/blob/master/lenacorners.png
