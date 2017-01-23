typedef struct Matcher Matcher;

typedef Image* (*findMatchesFunc)(Matcher*,Matrix*,Matrix*,Array*,Array*);

struct Matcher
{
  ImageUtil* imutil;
  MatrixUtil* matutil;
  findMatchesFunc findMatches;
};

Matcher* NewMatcher(ImageUtil* imutil);
