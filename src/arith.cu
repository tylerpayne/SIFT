#include <chai.h>

namespace chai {

// TUPLE ==
#define SET(K)                                                                 \
  template <> bool operator==(tuple<K> &lhs, tuple<K> &rhs) {                  \
    assert(lhs.length == rhs.length);                                          \
    for (int i = 0; i < lhs.length; i++) {                                     \
      if (lhs(i) != rhs(i))                                                    \
        return false;                                                          \
    }                                                                          \
    return true;                                                               \
  }

SET(char)
SET(int)
SET(float)
SET(double)

#undef SET

// ADD
#define SET(K)                                                                 \
  template <> matrix<K> operator+(matrix<K> &lhs, matrix<K> &rhs) {            \
    assert(lhs.shape == rhs.shape);                                            \
    matrix<K> ret(lhs.shape);                                                  \
    matrix<K>::memassert(lhs, DEVICE);                                         \
    matrix<K>::memassert(rhs, DEVICE);                                         \
    matrix<K>::memassert(ret, DEVICE);                                         \
    dim3 bdim, gdim;                                                           \
    make_launch_parameters(ret.shape, 1, &bdim, &gdim);                        \
    cuda::add_kernel<K><<<gdim, bdim, 0, lhs.stream>>>(                        \
        lhs.dev_ptr, rhs.dev_ptr, ret.dev_ptr, lhs.shape.prod());              \
    return ret;                                                                \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET
}
