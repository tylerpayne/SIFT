typedef struct Extractor Extractor;

typedef Array (*findCornerKeypointsFunc)(Extractor* self, Image*, int, float, float, int);

struct Extractor
{
  ImageUtil* imutil;
  Filters* filters;
  findCornerKeypointsFunc findCornerKeypoints;
};

Extractor* NewExtractor(ImageUtil* imutil);
