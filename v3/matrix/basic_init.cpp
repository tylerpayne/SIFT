#include <matrix.h>

template<typename K>
void Matrix<K>::basic_init(Tuple<int> s, bool isOnHost)
{
  shape = s;
  isHostSide = isOnHost;
}
