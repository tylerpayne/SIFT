typedef struct Camera Camera;

struct Camera
{
  float* K;
  float* R;
  Image* frame;
};
