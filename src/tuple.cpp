#include <chai.h>

namespace chai {

// CONSTRUCTORS
#define SET(K)                                                                 \
  template <> tuple<K>::tuple() {                                              \
    this->length = -1;                                                         \
    this->components = nullptr;                                                \
  }                                                                            \
  template <> tuple<K>::tuple(std::initializer_list<K> coords) {               \
    this->length = static_cast<int>(coords.size());                            \
    size_t size = sizeof(K) * this->length;                                    \
    this->components = static_cast<K *>(malloc(size));                         \
    memcpy((void *)(this->components), coords.begin(), size);                  \
  }                                                                            \
  template <> tuple<K>::tuple(K *components, int length) {                     \
    this->components = components;                                             \
    this->length = length;                                                     \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET

// FUNCTIONS
#define SET(K)                                                                 \
  template <> K tuple<K>::prod() {                                             \
    K retval = this->components[0];                                            \
    for (int i = 1; i < this->length; i++) {                                   \
      retval *= this->components[i];                                           \
    }                                                                          \
    return retval;                                                             \
  }                                                                            \
  template <> K tuple<K>::norm() {                                             \
    K retval = this->components[0];                                            \
    for (int i = 1; i < this->length; i++) {                                   \
      retval += this->components[i] * this->components[i];                     \
    }                                                                          \
    return sqrt(retval);                                                       \
  }                                                                            \
  template <> K tuple<K>::norm(int l) {                                        \
    K retval = this->components[0];                                            \
    for (int i = 1; i < this->length; i++) {                                   \
      retval += pow(this->components[i], l);                                   \
    }                                                                          \
    return pow(retval, 1 / static_cast<K>(l));                                 \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET

// OPERATORS
#define SET(K)                                                                 \
  template <> K tuple<K>::operator()(int i) { return this->components[i]; }    \
  template <> tuple<K>::operator int_tuple() {                                 \
    tuple<int> t = static_cast<tuple<int>>(*this);                             \
    int_tuple ret;                                                             \
    ret.length = t.length;                                                     \
    ret.components = (int *)malloc(sizeof(int) * ret.length);                  \
    memcpy(ret.components, t.components, sizeof(int) * ret.length);            \
    return ret;                                                                \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET
}
