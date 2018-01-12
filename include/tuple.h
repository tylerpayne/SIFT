namespace chai {
typedef struct int_tuple { int *components, length; } int_tuple;

template <typename K> class tuple {
public:
  // VARIABLES
  K *components;
  int length;

  // CONSTRUCTORS
  tuple();
  tuple(std::initializer_list<K> coords);
  tuple(K *components, int length);

  // FUNCTIONS
  K prod();
  K norm();
  K norm(int l);

  // OPERATORS
  K operator()(int i);

  #define SET(OP) tuple<K>& operator OP(tuple<K> &rhs);

  SET(+=)
  SET(-=)
  SET(*=)
  #undef SET

  #define SET(OP) tuple<K>& operator OP(K rhs);

  SET(+=)
  SET(-=)
  SET(*=)
  SET(/=)
  #undef SET

  // CONVERSION OPERATORS
  operator int_tuple();
  template <typename F> operator tuple<F>() {
    F *comp = static_cast<F *>(malloc(sizeof(F) * (this->length)));
    for (int i = 0; i < this->length; i++) {
      comp[i] = static_cast<F>(this->components[i]);
    }
    tuple<F> t(comp, this->length);
    return t;
  }
};
}
