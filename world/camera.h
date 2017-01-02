typedef struct Camera Camera;

struct Camera
{
  Matrix* K;
  Matrix* R;
  Image* frame;
};
