# computervision (UNDER DEVELOPMENT)

A collection of computer vision, data structures, and math utilites all implemented in C and accelerated with [CUDA v8.0].

## Core Components:

### [MatrixUtil]
The abstract definition of the MatrixUtil class, allowing concrete implementations for different platforms (i.e. CUDA, Metal)
#### [PrimitiveMatrixUtil]
A plain-C implementation of the MatrixUtil.

#### [CUDAMatrixUtil] 
(INCOMPLETE)
A [CUBLAS] and [CUSOLVER] implementation of the MatrixUtil.

#### TODO: MetalMatrixUtil
For now, see my other repo [MetalUnity].

### [ImageUtil]

##### File Handling
(only PNGs are supported currently)
All file handling is currently done with [lodepng] 

##### NPP
Currently, the ImageUtil class is a concrete implementation, unlike the MatrixUtil class. It uses the [Nvidia Performance Primitives] library to perform common arithmetic, geometric and statistical operations on Images.

ToDo: Abstract this class and use MatrixUtil as lowlevel image representations, not Npp32f.

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
