
namespace chai
{
    template <typename K>
    Tuple<K>::Tuple()
    {
      this->length=-1;
      this->components=NULL;
    }

    template <typename K>
    Tuple<K>::Tuple(Tuple<K> *t)
    {
      this->length = t->length;
      size_t size = sizeof(K)*this->length;
      this->components = static_cast<K*>(malloc(size));
      memcpy((void *)(this->components), t->components, size);
    }

    template<typename K>
    Tuple<K>::Tuple(std::initializer_list<K> coords)
    {
      this->length = static_cast<int>(coords.size());
      size_t size = sizeof(K)*this->length;
      this->components = static_cast<K*>(malloc(size));
      memcpy((void *)(this->components), coords.begin(), size);
    }

    template <typename K>
    K Tuple<K>::prod()
    {
      K retval = this->components[0];
      for (int i = 1; i < this->length; i++)
      {
        retval *= this->components[i];
      }
      return retval;
    }

    template <typename K>
    K Tuple<K>::norm()
    {
      K retval = this->components[0];
      for (int i = 1; i < this->length; i++)
      {
        retval += this->components[i]* this->components[i];
      }
      return sqrt(retval);
    }

    template <typename K>
    K Tuple<K>::norm(int l)
    {
      K retval = this->components[0];
      for (int i = 1; i < this->length; i++)
      {
        retval += pow(this->components[i],l);
      }
      return pow(retval,1/(K)l);
    }

    template <typename K>
    K Tuple<K>::operator()(int i)
    {
        return this->components[i];
    }
}
