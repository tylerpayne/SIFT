# computervision (UNDER DEVELOPMENT)

A collection of computer vision, data structures, and math utilities all implemented in C and accelerated with [CUDA v8.0].

# Design Notes
This library is designed with an object oriented architecture. This design avoids cryptic function names and abstracts all the computer vision specific methods away from the math and image utilities. Thus the library is platform-agnostic; everything in the library refers to compiled versions of the MatrixUtil and ImageUtil, which can be built for any platform (i.e. CUDA, OpenCL, Metal, etc.).

A minor annoyance of using this design in the C language is the need to pass a `self` argument to functions. 
(e.g. `imutil->convolve(imutil,image,kernel)`)

# Core Components:

## Core.h
A collection of definitions, typedefs and helpful functions that the entire library depends upon.

## [MatrixUtil]
The abstract definition of the MatrixUtil class, allowing concrete implementations for different platforms (i.e. CUDA, Metal)
### [PrimitiveMatrixUtil]
A plain-C implementation of the MatrixUtil.

### [CUDAMatrixUtil]
The [CUBLAS] and [CUSOLVER] implementation of the MatrixUtil.

### TODO: MetalMatrixUtil
For now, see my other repo [MetalUnity].

## [ImageUtil]
The abstract definition of the ImageUtil class, allowing concrete implementations for different platforms (i.e. CUDA, Metal). ImageUtil depends on MatrixUtil's Matrix initializers for the low level representations of images.

### [CUDAImageUtil]
The CUDA implementation of the ImageUtil.

##### NPP
 CUDAImageUtil uses the [Nvidia Performance Primitives] library to perform common arithmetic, geometric and statistical operations on Images that are not already supported by the underlying MatrixUtil.
 
 ## [IOUtil]
 The I/O Utility handles file management e.g. loading images.
 
 Currently suppported image formats: JPEG,PNG,BMP,ICO
 
 ## [DrawUtil]
 The Draw Utility uses [GTK+] v2 to display images and draw elements like keypoints on top of them.

# Data Structures

# Computer Vision Components

## [Filters]

## [Extractor]

## [Matcher]

# Example
```C
  int gw = 8; // Width of Gaussian Kernels
  int g1s = 5; // Sigma of Gaussian Kernel 1
  int g2s = 3; // Sigma of Gaussian Kernel 2
  int mw = 15; // Width of LocalMax window
  char* saves = "DoG.png"; // Filepath to save to
  char* path = "image.png"; // Filepath to load from
  //Init the utilities
  MatrixUtil* matutil = GetMatrixUtil(); // Get CUDA MatrixUtil
  ImageUtil* imutil = GetImageUtil(matutil); // Get CUDA Image Util
  IOUtil* ioutil = GetIOUtil(imutil);
  //Init the CV objects
  Filters *filters = GetFilters(imutil);
  //Load the image
  Image* in = ioutil->loadImageFromFile(ioutil,path);
  Image* im = imutil->resample(imutil,in,256,256);
  //Create the two gaussians
  Image* gauss1 = filters->makeGaussianKernel(imutil,gw,g1s);
  Image* gauss2 = filters->makeGaussianKernel(imutil,gw,g2s);
  //Get difference of gaussian kernel
  Image* DoGKernel = imutil->subtract(imutil,gauss1,gauss2);
  //Convolve
  Image* DoGImage = imutil->convolve(imutil,im,DoGKernel);
  //Find corners (local maximums)
  Image* corners = imutil->max(imutil,DoGImage,mw);
  //Save image
  ioutil->saveImageToFile(ioutil,corners,saves); // This is the only function that copies memory from device to host
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
[IOUtil]:https://github.com/tylerpayne/computervision/blob/master/utils/IOUtil.h
[DrawUtil]:https://github.com/tylerpayne/computervision/blob/master/utils/DrawUtil.h
[lena]:https://github.com/tylerpayne/computervision/blob/master/lena256.png
[corners]:https://github.com/tylerpayne/computervision/blob/master/lenacorners.png
