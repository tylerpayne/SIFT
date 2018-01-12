namespace chai {

#define SET(OP)                                                                \
  template <typename K> matrix<K> operator OP(matrix<K> &lhs, matrix<K> &rhs); \
  template <typename K> tuple<K> operator OP(tuple<K> &lhs, tuple<K> &rhs);

SET(+)
SET(-)
SET(*)
#undef SET

#define SET(OP)                                                                \
  template <typename K> matrix<K> operator OP(matrix<K> &lhs, K rhs); \
  template <typename K> tuple<K> operator OP(tuple<K> &lhs, K rhs);

SET(+)
SET(-)
SET(*)
SET(/)

#undef SET

#define SET(OP)                                                                \
  template <typename K> matrix<K> operator OP(matrix<K> &lhs, matrix<K> &rhs);

SET(<)
SET(>)
SET(<=)
SET(>=)
SET(==)
#undef SET

template <typename K> bool operator==(tuple<K> &lhs, tuple<K> &rhs);
}
