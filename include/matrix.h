
namespace chai {

  template<typename K>
  void matrix<K>::basic_init(matrix<K> *m, tuple<int> &s, bool isOnHost)
  {
    m->shape = s;
    m->isHostSide = true;
  }

  template<typename K>
  void matrix<K>::empty_init(matrix<K> *m, tuple<int> &s, bool isOnHost)
  {
    basic_init(m,s,true);
    if (m->isHostSide)
    {
      size_t sz = sizeof(K)*m->shape.prod();
      m->host_ptr = (K*)malloc(sz);
      memset((void *)(m->host_ptr),0,sz);
      m->dev_ptr = NULL;
    } else
    {
      size_t sz = sizeof(K)*m->shape.prod();
      cudaMalloc((void**)&(m->dev_ptr),sz);
      cudaMemset((void*)(m->dev_ptr),0,sz);
      m->host_ptr = NULL;
    }
  }

  template<typename K>
  matrix<K>::matrix(tuple<int> &s)
  {
    empty_init(this,s,true);
    printf("returning matrix\n");
  }

  template<typename K>
  matrix<K>::matrix(std::initializer_list<int> s)
  {
    tuple<int> t(s);
    empty_init(this,t,true);
  }

  template<typename K>
  matrix<K>::matrix(K* ptr, bool isHostPtr, tuple<int> &s)
  {
    basic_init(s,isHostPtr);
    if (isHostPtr)
    {
      this->host_ptr = ptr;
      this->dev_ptr = NULL;
    } else
    {
      this->dev_ptr = ptr;
      this->host_ptr = NULL;
    }
  }

  template<typename K>
  matrix<K>::matrix(K c, bool onHost, tuple<int> &s)
  {
    empty_init(s,onHost);
    if (onHost)
    {
      K *tmp_ptr = this->host_ptr;
      for (int i = 0; i < this->shape.prod(); i++)
      {
        *tmp_ptr = c;
        tmp_ptr++;
      }
    } else
    {
      //CUDA FILL KERNEL
    }
  }

  template <typename K>
  matrix<K>::~matrix()
  {
    if (this->isHostSide)
    {
      free(this->host_ptr);
    } else
    {
      cuda::safe_call<cudaError_t>(cudaFree(this->dev_ptr));
    }
    this->host_ptr = NULL;
    this->dev_ptr = NULL;
  }

  template <typename K>
  matrix<K> matrix<K>::operator()(std::initializer_list<int> rows, std::initializer_list<int> cols)
  {
    tuple<int> r(rows);
    tuple<int> c(cols);
    return (*this)(r,c);
  }

  template <typename K>
  matrix<K> matrix<K>::operator()(tuple<int> &rows, tuple<int> &cols)
  {
    matrix ret({rows.length, cols.length});
    //COPY!
    if (this->isHostSide)
    {

    } else
    {

    }
    return ret;
  }

  template <typename K>
  matrix<K> matrix<K>::operator+(matrix<K> m)
  {
    memassert(this,DEVICE);
    memassert(m,DEVICE);

    matrix<K> ret(this->shape);

    memassert(ret,DEVICE);

    dim3 bdim,gdim;
    make_launch_parameters(this->shape,1,&bdim,&gdim);
    cuda::add_kernel<K><<<gdim,bdim,0,stream>>>(this->dev_ptr,m.dev_ptr,ret.dev_ptr,this->shape);
    return ret;
  }

/*

  // ########## //
  // OPERATIONS //
  // ########### //


  //############//
  // DESTRUCTOR //
  //############//

  extern cudaStream_t _cudaStream;
  template<typename K>
  void matrix<K>::memassert(int dest)
  {
    if (this->isHostSide != dest)
    {
      size_t size = sizeof(K)*this->shape.prod();
      if (dest == DEVICE)
      {
        K *new_dev_ptr;
        cudaMalloc((void**)&(new_dev_ptr),size);
        cudaMemcpyAsync(dev_ptr,this->host_ptr,size,cudaMemcpyHostToDevice,_cudaStream);
        this->dev_ptr = dev_ptr;
        this->isHostSide = false;
      }
      else if (dest == HOST)
      {
        cudaMemcpyAsync(this->host_ptr,this->dev_ptr,size,cudaMemcpyDeviceToHost,_cudaStream);
        cudaFree(this->dev_ptr);
        this->dev_ptr = NULL;
        this->isHostSide = true;
      }
    }
  }
*/
}
