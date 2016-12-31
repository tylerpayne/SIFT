#include "world/camera.c"

typedef struct StochasticEpipolarAligner StochasticEpipolarAligner;

struct StochasticEpipolarAligner
{
  Camera* camLeft;
  Camera* camRight;
};
