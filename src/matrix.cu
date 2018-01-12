#include <chai.h>

namespace chai {

// STATIC FUNCTIONS
#define SET(K)                                                                 \
  template <>                                                                  \
  void matrix<K>::basic_init(matrix<K> &m, tuple<int> s, bool isOnHost) {      \
    m.shape = s;                                                               \
    m.isHostSide = isOnHost;                                                   \
    m.stream = 0;                                                              \
  }                                                                            \
  template <>                                                                  \
  void matrix<K>::empty_init(matrix<K> &m, tuple<int> s, bool isOnHost) {      \
    basic_init(m, s, isOnHost);                                                \
    size_t sz = sizeof(K) * m.shape.prod();                                    \
    if (m.isHostSide) {                                                        \
      m.host_ptr = static_cast<K *>(malloc(sz));                               \
      memset((void *)(m.host_ptr), 0, sz);                                     \
      m.dev_ptr = nullptr;                                                     \
    } else {                                                                   \
      K *d_ptr;                                                                \
      cudaMalloc((void **)&(d_ptr), sz);                                       \
      cudaMemset((void *)(d_ptr), 0, sz);                                    \
      m.dev_ptr = d_ptr;                                                       \
      m.host_ptr = nullptr;                                                    \
    }                                                                          \
  }                                                                            \
  template <> void matrix<K>::memassert(matrix<K> &m, int dest) {              \
    printf("memasserting: %i==%i\n", m.isHostSide, dest);                      \
    if (m.isHostSide == dest)                                                  \
      return;                                                                  \
    K *from, *to;                                                              \
    enum cudaMemcpyKind direction;                                             \
    size_t sz = sizeof(K) * m.shape.prod();                                    \
    if (dest == DEVICE) {                                                      \
      from = m.host_ptr;                                                       \
      cuda::safe_call<cudaError_t>(cudaMalloc((void **)&to, sz));              \
      m.dev_ptr = to;                                                          \
      direction = cudaMemcpyHostToDevice;                                      \
    } else {                                                                   \
      from = m.dev_ptr;                                                        \
      direction = cudaMemcpyDeviceToHost;                                      \
      to = static_cast<K *>(malloc(sz));                                       \
      m.host_ptr = to;                                                         \
    }                                                                          \
    cuda::safe_call<cudaError_t>(cudaMemcpy(to, from, sz, direction));         \
    m.isHostSide = dest;                                                       \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET

// CONSTRUCTORS
#define SET(K)                                                                 \
  template <> matrix<K>::matrix(std::initializer_list<int> s) {                \
    tuple<int> t(s);                                                           \
    empty_init(*this, t, true);                                                \
  }                                                                            \
  template <> matrix<K>::matrix(tuple<int> s) { empty_init(*this, s, true); }  \
  template <>                                                                  \
  matrix<K>::matrix(K *ptr, bool isHostPtr, std::initializer_list<int> s) {    \
    tuple<int> t(s);                                                           \
    basic_init(*this, t, isHostPtr);                                           \
    if (isHostPtr) {                                                           \
      this->host_ptr = ptr;                                                    \
      this->dev_ptr = nullptr;                                                 \
    } else {                                                                   \
      this->dev_ptr = ptr;                                                     \
      this->host_ptr = nullptr;                                                \
    }                                                                          \
  }                                                                            \
  template <> matrix<K>::matrix(K *ptr, bool isHostPtr, tuple<int> s) {        \
    basic_init(*this, s, isHostPtr);                                           \
    if (isHostPtr) {                                                           \
      this->host_ptr = ptr;                                                    \
      this->dev_ptr = nullptr;                                                 \
    } else {                                                                   \
      this->dev_ptr = ptr;                                                     \
      this->host_ptr = nullptr;                                                \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  matrix<K>::matrix(K c, bool onHost, std::initializer_list<int> s) {          \
    tuple<int> t(s);                                                           \
    empty_init(*this, t, onHost);                                              \
    if (onHost) {                                                              \
      for (int i = 0; i < this->shape.prod(); i++) {                           \
        this->host_ptr[i] = c;                                                 \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  template <> matrix<K>::matrix(K c, bool onHost, tuple<int> s) {              \
    empty_init(*this, s, onHost);                                              \
    if (onHost) {                                                              \
      for (int i = 0; i < this->shape.prod(); i++) {                           \
        this->host_ptr[i] = c;                                                 \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  template <> matrix<K>::~matrix() {                                           \
    free(this->host_ptr);                                                      \
    cuda::safe_call<cudaError_t>(cudaFree(this->dev_ptr));                     \
    this->host_ptr = nullptr;                                                  \
    this->dev_ptr = nullptr;                                                   \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET

// OPERATORS
#define SET(K)                                                                 \
  template <> K matrix<K>::operator()(tuple<int> index) {                      \
    assert(index.length == 2);                                                 \
    assert((index(0) < this->shape(0)) && index(1) < this->shape(1));          \
    matrix<K>::memassert(*this, HOST);                                         \
    K ret;                                                                     \
    memcpy(&ret, this->host_ptr + (index(0) * this->shape(0)) + index(1),      \
           sizeof(K));                                                         \
    return ret;                                                                \
  }                                                                            \
  template <> K matrix<K>::operator()(std::initializer_list<int> index) {      \
    tuple<int> t(index);                                                       \
    return (*this)(t);                                                         \
  }                                                                            \
  template <>                                                                  \
  matrix<K> matrix<K>::operator()(tuple<int> rows, tuple<int> cols) {          \
    matrix ret({rows.length, cols.length});                                    \
    if (this->isHostSide) {                                                    \
      for (int i = 0; i < rows.length; i++) {                                  \
        memcpy(ret.host_ptr, this->host_ptr, sizeof(K) * cols.length);         \
      }                                                                        \
    }                                                                          \
    return ret;                                                                \
  }                                                                            \
  template <>                                                                  \
  matrix<K> matrix<K>::operator()(std::initializer_list<int> rows,             \
                                  std::initializer_list<int> cols) {           \
    tuple<int> r(rows);                                                        \
    tuple<int> c(cols);                                                        \
    return (*this)(r, c);                                                      \
  }                                                                            \
  template <> K matrix<K>::operator()(int row, int col) {                      \
    if (this->isHostSide) {                                                    \
      int c = idx2c(row, col, this->shape);                                    \
      return this->host_ptr[c];                                                \
    } else {                                                                   \
      return static_cast<K>(0);                                                \
    }                                                                          \
  }                                                                            \
  template <> matrix<K> &matrix<K>::operator+=(matrix<K> &rhs) {               \
    assert(this->shape == rhs.shape);                                          \
    matrix<K>::memassert(*this, DEVICE);                                       \
    matrix<K>::memassert(rhs, DEVICE);                                         \
    dim3 bdim, gdim;                                                           \
    make_launch_parameters(this->shape, 1, &bdim, &gdim);                      \
    cuda::add_kernel<K><<<gdim, bdim, 0, this->stream>>>(                      \
        this->dev_ptr, rhs.dev_ptr, this->dev_ptr, this->shape.prod());        \
    return *this;                                                              \
  }                                                                            \
  template <> matrix<K> &matrix<K>::operator-=(matrix<K> &rhs) {               \
    assert(this->shape == rhs.shape);                                          \
    matrix<K>::memassert(*this, DEVICE);                                       \
    matrix<K>::memassert(rhs, DEVICE);                                         \
    dim3 bdim, gdim;                                                           \
    make_launch_parameters(this->shape, 1, &bdim, &gdim);                      \
    cuda::subtract_kernel<K><<<gdim, bdim, 0, this->stream>>>(                 \
        this->dev_ptr, rhs.dev_ptr, this->dev_ptr, this->shape.prod());        \
    return *this;                                                              \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET
}
