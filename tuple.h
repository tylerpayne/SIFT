
namespace chai {


//////////////////
// CONSTRUCTORS //
//////////////////

template <typename K> tuple<K>::tuple() {
  this->length = -1;
  this->components = NULL;
}

template <typename K> tuple<K>::tuple(std::initializer_list<K> coords) {
  this->length = static_cast<int>(coords.size());
  size_t size = sizeof(K) * this->length;
  this->components = static_cast<K *>(malloc(size));
  memcpy((void *)(this->components), coords.begin(), size);
}

template <> tuple<K>::tuple(K *components, int length)
{
  this->components = components;
  this->length = length;
}

///////////////
// FUNCTIONS //
//////////////

template <> K tuple<K>::prod() {
  K retval = this->components[0];
  for (int i = 1; i < this->length; i++) {
    retval *= this->components[i];
  }
  return retval;
}

template <> K tuple<K>::norm() {
  K retval = this->components[0];
  for (int i = 1; i < this->length; i++) {
    retval += this->components[i] * this->components[i];
  }
  return sqrt(retval);
}

template <> K tuple<K>::norm(int l) {
  K retval = this->components[0];
  for (int i = 1; i < this->length; i++) {
    retval += pow(this->components[i], l);
  }
  return pow(retval, 1 / (K)l);
}

///////////////
// OPERATORS //
///////////////

template <typename K> K tuple<K>::operator()(int i) {
  return this->components[i];
}

template <typename K> bool tuple<K>::operator==(tuple<K> t) {
  if (t.length != this->length)
    return false;
  for (int i = 0; i < this->length; i++) {
    if ((*this)(i) != t(i))
      return false;
  }
  return true;
}

template <typename K> tuple<K>::operator int_tuple() {
  int_tuple ret;
  ret.length = this->length;
  ret.components = this->components;
  return ret;
}
}
