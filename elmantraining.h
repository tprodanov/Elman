#pragma once

#include "elmannetwork.h"
//#include "trainingnetwork.h"

class ElmanNetwork;

class ElmanTraining
{
public:
	ElmanTraining(int, double*, double*, int, int ,int);
	ElmanTraining(int, double*, int*, int, int, int);
	void Train (int, int, double);
	void saveWeigths (char*);
private:
	void makeDeltas (int);
	void changeWeights (double);
	double *mXV, *mTV, *mNowX;
	int mXS, mHS, mYS, mNumExamples;
	ElmanNetwork *mNetwork;
	double *mDeltaH, *mDeltaY;
};
