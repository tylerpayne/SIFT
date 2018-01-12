namespace chai {

template <typename K> class matrix {
public:
  bool isHostSide, T;
  K *host_ptr, *dev_ptr;
  tuple<int> shape;
  cudaStream_t stream;

  static void basic_init(matrix<K> &m, tuple<int> s, bool isOnHost);
  static void empty_init(matrix<K> &m, tuple<int> s, bool isOnHost);
  static void const_init(matrix<K> &m, K val, tuple<int> s, bool isOnHost);

  static void memassert(matrix<K> &m, int dest);

  matrix(std::initializer_list<int> s);
  matrix(tuple<int> s);

  matrix(K *ptr, bool isHostPtr, std::initializer_list<int> s);
  matrix(K *ptr, bool isHostPtr, tuple<int> s);

  matrix(K c, bool onHost, std::initializer_list<int> s);
  matrix(K c, bool onHost, tuple<int> s);

  ~matrix();

  K operator()(int row, int col);
  K operator()(tuple<int> index);
  K operator()(std::initializer_list<int> index);
  matrix operator()(std::initializer_list<int> rows,
                    std::initializer_list<int> cols);
  matrix operator()(tuple<int> rows, tuple<int> cols);

#define SET(OP) matrix<K> &operator OP(matrix<K> &rhs);

  SET(+=)
  SET(-=)
  SET(*=)
#undef SET

#define SET(OP) matrix<K> &operator OP(K rhs);

  SET(+=)
  SET(-=)
  SET(*=)
  SET(/=)
#undef SET

  template <typename F> operator matrix<F>() {
    if (this->isHostSide) {
      int sz = this->shape.prod();
      F *h_ptr = static_cast<F *>(malloc(sizeof(F) * sz));
      for (int i = 0; i < sz; i++) {
        h_ptr[i] = static_cast<F>(this->host_ptr[i]);
      }
      matrix<F> m(h_ptr, true, this->shape);
      return m;
    } else {
      // cuda conversion kernel.
    }
  }

  void print() {
    memassert(*this, HOST);
    for (int i = 0; i < this->shape(0); i++) {
      for (int j = 0; j < this->shape(1); j++) {
        std::cout << "[ " << (*this)(i, j) << " ]";
      }
      std::cout << "\n";
    }
  }

  void toDevice() { memassert(*this, DEVICE); }
  void toHost() { memassert(*this, HOST); }
};
}
