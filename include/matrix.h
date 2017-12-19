
namespace chai {

  //////////////////
  // STATIC FUNCS //
  //////////////////

  template<typename K>
  void matrix<K>::basic_init(matrix<K> &m, tuple<int> s, bool isOnHost)
  {
    m.shape = s;
    m.isHostSide = isOnHost;
  }

  template<typename K>
  void matrix<K>::empty_init(matrix<K> &m, tuple<int> s, bool isOnHost)
  {
    basic_init(m,s,isOnHost);
    if (m.isHostSide)
    {
      size_t sz = sizeof(K)*m.shape.prod();
      m.host_ptr = (K*)malloc(sz);
      memset((void *)(m.host_ptr),0,sz);
      m.dev_ptr = NULL;
    } else
    {
      size_t sz = sizeof(K)*m.shape.prod();
      cudaMalloc((void**)&(m.dev_ptr),sz);
      cudaMemset((void*)(m.dev_ptr),0,sz);
      m.host_ptr = NULL;
    }
  }

  template<typename K>
  void matrix<K>::memassert(matrix<K> &m, int dest)
  {
    if (m.isHostSide == dest) return;

    K *from, *to;
    enum cudaMemcpyKind direction;
    size_t sz = sizeof(K)*m.shape.prod();

    if (dest == DEVICE)
    {
      from = m.host_ptr;
      to = m.dev_ptr;
      direction = cudaMemcpyHostToDevice;

      cudaMalloc((void**)&to,sz);

    }
    else
    {
      from = m.dev_ptr;
      to = m.host_ptr;
      direction = cudaMemcpyDeviceToHost;
      to = (K*)malloc(sz);

    }
    cudaMemcpy(to,from,sz,direction);
  }

  //////////////////
  // CONSTRUCTORS //
  //////////////////

  template<typename K>
  matrix<K>::matrix(std::initializer_list<int> s)
  {
    tuple<int> t(s);
    empty_init(*this,t,true);
  }

  template<typename K>
  matrix<K>::matrix(tuple<int> s)
  {
    empty_init(*this,s,true);
  }

  template<typename K>
  matrix<K>::matrix(K* ptr, bool isHostPtr, std::initializer_list<int> s)
  {
    tuple<int> t(s);
    basic_init(*this,t,isHostPtr);
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
  matrix<K>::matrix(K* ptr, bool isHostPtr, tuple<int> s)
  {
    basic_init(this,s,isHostPtr);
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
  matrix<K>::matrix(K c, bool onHost, std::initializer_list<int> s)
  {
    tuple<int> t(s);
    empty_init(this,t,onHost);
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

  template<typename K>
  matrix<K>::matrix(K c, bool onHost, tuple<int> s)
  {
    empty_init(this,s,onHost);
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

  ////////////////
  // DESTRUCTOR //
  ////////////////

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

  ///////////////
  // OPERATORS //
  //////////////

  template <typename K>
  K matrix<K>::operator()(tuple<int> index)
  {
    assert(index.length == 2);
    assert((index(0) < this->shape(0)) && index(1) < this->shape(1));
    matrix<K>::memassert(*this,HOST);
    K ret;
    memcpy(&ret,this->host_ptr+(index(0)*this->shape(0))+index(1),sizeof(K));
    return ret;
  }

  template <typename K>
  K matrix<K>::operator()(std::initializer_list<int> index)
  {
    tuple<int> t(index);
   return (*this)(t);
  }

  template <typename K>
  matrix<K> matrix<K>::operator()(tuple<int> rows, tuple<int> cols)
  {
    matrix ret({rows.length, cols.length});
    //COPY!
    if (this->isHostSide)
    {
      for (int i = 0; i < rows.length; i++)
      {
        memcpy(ret.host_ptr,this->host_ptr,sizeof(K)*cols.length);
      }
    } else
    {
      //CUDA COPY KERNEL!
    }
    return ret;
  }

  //()

  template <typename K>
  matrix<K> matrix<K>::operator()(std::initializer_list<int> rows, std::initializer_list<int> cols)
  {
    tuple<int> r(rows);
    tuple<int> c(cols);
    return (*this)(r,c);
  }

  //+

  template <typename K>
  matrix<K> matrix<K>::operator+(matrix<K> &m)
  {
    assert(m.shape == this->shape);
    matrix<K>::memassert(*this,DEVICE);
    matrix<K>::memassert(m,DEVICE);

    matrix<K> ret(this->shape);

    matrix<K>::memassert(ret,DEVICE);

    dim3 bdim,gdim;
    make_launch_parameters(this->shape,1,&bdim,&gdim);
    cuda::add_kernel<K><<<gdim,bdim,0,stream>>>(this->dev_ptr,m.dev_ptr,ret.dev_ptr,(int_tuple)(this->shape));
    return ret;
  }

  template <typename K>
  matrix<K> matrix<K>::operator+(K c)
  {
    matrix<K>::memassert(*this,DEVICE);

    matrix<K> ret(this->shape);

    matrix<K>::memassert(ret,DEVICE);

    dim3 bdim,gdim;
    make_launch_parameters(this->shape,1,&bdim,&gdim);
    cuda::addc_kernel<K><<<gdim,bdim,0,stream>>>(this->dev_ptr,c,ret.dev_ptr,(int_tuple)(this->shape));
    return ret;
  }

  //-

  template <typename K>
  matrix<K> matrix<K>::operator-(matrix<K> &m)
  {
    assert(m.shape == this->shape);
    matrix<K>::memassert(*this,DEVICE);
    matrix<K>::memassert(m,DEVICE);

    matrix<K> ret(this->shape);

    matrix<K>::memassert(ret,DEVICE);

    dim3 bdim,gdim;
    make_launch_parameters(this->shape,1,&bdim,&gdim);
    cuda::subtract_kernel<K><<<gdim,bdim,0,stream>>>(this->dev_ptr,m.dev_ptr,ret.dev_ptr,(int_tuple)(this->shape));
    return ret;
  }

  template <typename K>
  matrix<K> matrix<K>::operator-(K c)
  {
    matrix<K>::memassert(*this,DEVICE);

    matrix<K> ret(this->shape);

    matrix<K>::memassert(ret,DEVICE);

    dim3 bdim,gdim;
    make_launch_parameters(this->shape,1,&bdim,&gdim);
    cuda::subtractc_kernel<K><<<gdim,bdim,0,stream>>>(this->dev_ptr,c,ret.dev_ptr,(int_tuple)(this->shape));
    return ret;
  }

  /////////
  // ETC //
  /////////

  template<typename K>
  void matrix<K>::print()
  {
    matrix<K> m = *this;
    for (int i = 0; i < m.shape(0); i++)
    {
      printf("[ %3i ]", i);    typedef struct int_tuple
    {
      int *components, length;
    } int_tuple;
      for (int j = 0; j < m.shape(1); j++)
      {
        printf("  %d  ",m({i,j}));
      }
      printf("\n");
    }
  }
}
