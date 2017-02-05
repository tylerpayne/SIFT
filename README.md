# computervision
####(UNDER DEVELOPMENT)

A collection of math utilities , data structures, and computer vision methods and all implemented in C and accelerated with [CUDA v8.0].

# Example
```C
  #include <utils/ImageUtil.h>
  #include <generators/Filters.h>

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
  Filters *filters = NewFilters(imutil);
  //Load the image
  Image* in = ioutil->loadImageFromFile(ioutil,path);
  Image* im = imutil->resample(imutil,in,256,256);
  //Create the two gaussians
  Image* gauss1 = filters->makeGaussianKernel(filters,gw,g1s);
  Image* gauss2 = filters->makeGaussianKernel(filters,gw,g2s);
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
[CUDA v8.0]:https://developer.nvidia.com/cuda-toolkit


[lena]:https://github.com/tylerpayne/computervision/blob/master/lena256.png
[corners]:https://github.com/tylerpayne/computervision/blob/master/lenacorners.png
