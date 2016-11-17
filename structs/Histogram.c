#include "Histogram.h"

void addImpl(Histogram* self, float val)
{
  float perBinRange = self->range / (float)self->nbins;
  for (int i =0; i < self->nbins; i++)
  {
    if (val > (i*perBinRange) && val <= (i*perBinRange)+perBinRange)
    {
      ((LinkedList*)self->bins->get(self->bins,i)->value)->add(NewListNode((void*)&val));
      self->binTotals[i] += val;
      break;
    }
  }
}

void addTrilinearInterpolateImpl(Histogram* self, float val)
{
  float perBinRange = self->range / (float)self->nbins;
  for (int i =0; i < self->nbins; i++)
  {
    if (val > (i*perBinRange) && val <= (i*perBinRange)+perBinRange)
    {
      ((LinkedList*)self->bins->get(self->bins,i)->value)->add(NewListNode((void*)&val));
      float rbinSpill = ((val - (perBinRange/2.0))/(perBinRange/2.0))+0.5;
      float lbinSpill = 1 - rbinSpill;
      rbinSpill = rbinSpill*val;
      lbinSpill = lbinSpill*val;
      if (i > 0)
      {
          ((LinkedList*)self->bins->get(self->bins,i-1)->value)->add(NewListNode((void*)&lbinSpill));
      }
      if (i < self->nbins - 1)
      {
          ((LinkedList*)self->bins->get(self->bins,i+1)->value)->add(NewListNode((void*)&rbinSpill));
      }
      self->binTotals[i] += val;
      self->binTotals[i-1] += lbinSpill;
      self->binTotals[i+1] += rbinSpill;
      break;
    }
  }
}

int maxBinImpl(Histogram* self)
{
  float maxVal = 0;
  int maxIdx = -1;
  for (int i = 0; i < self->nbins; i++)
  {
    if (self->binTotals[i] > maxVal)
    {
      maxVal = self->binTotals[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

int minBinImpl(Histogram* self)
{
  float minVal = 0;
  int minIdx = -1;
  for (int i = 0; i < self->nbins; i++)
  {
    if (self->binTotals[i] < minVal)
    {
      minVal = self->binTotals[i];
      minIdx = i;
    }
  }
  return minIdx;
}

void tossOutliersImpl(Histogram* self, int groupSize, float errMargin)
{

}

Histogram* NewHistogram(float range, int nbins)
{
  Histogram* histogram = (Histogram*)malloc(sizeof(Histogram));
  histogram->binRange = range;
  histogram->nbins = nbins;
  histogram->binTotals = *float[nbins];
  histogram->bins = NewLinkedList();
  for (int i = 0; i < nbins; i++)
  {
    LinkedList* bin = NewLinkedList();
    ListNode* n = NewListNode((void*)bin);
    histogram->bins->append(histogram->bins,n);
  }

  histogram->add = addImpl;
  histogram->addTrilinearInterpolate = addTrilinearInterpolateImpl;
  histogram->maxBin = maxBinImpl;
  histogram->minBin = minBinImpl;
  histogram->tossOutliers = tossOutliersImpl;

  return histogram;
}
