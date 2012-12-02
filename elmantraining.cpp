#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fstream"

#include <iostream>

#include "elmantraining.h"

ElmanTraining::ElmanTraining(int NumExamples, double *XVec, double *TVec, int n, int k, int m)
{
	mXS = n;
	mHS = k;
	mYS = m;
	mNumExamples = NumExamples;
	mXV = new double [mNumExamples * mXS];
	mTV = new double [mNumExamples * mYS];
	int i, j;
	for (i = 0; i < mNumExamples; i++) {
		for (j = 0; j < mXS; j++) {
			mXV [mXS * i + j] = XVec [mXS * i + j];
		}
		for (j = 0; j < mYS; j++) {
			mTV [mYS * i + j] = TVec [mYS * i + j];
		}
	}
	double *mNowX;
	mNowX = new double [mXS];
	for (int i = 0; i < mXS; i++)
		mNowX [i] = XVec [i];
	mNetwork = new ElmanNetwork (n, k, m, mNowX);
	mDeltaH = new double [mHS];
	mDeltaY = new double [mYS];
}

ElmanTraining::ElmanTraining(int NumExamples, double *XVec, int *TVecChar, int n, int k, int m)
{
	mXS = n;
	mHS = k;
	mYS = m;
	mNumExamples = NumExamples;
	mXV = new double [mNumExamples * mXS];
	mTV = new double [mNumExamples * mYS];
	int i, j;
	for (i = 0; i < mNumExamples; i++) {
		for (j = 0; j < mXS; j++) {
			mXV [mXS * i + j] = XVec [mXS * i + j];
		}
		for (j = 0; j < mYS; j++) {
			mTV [mYS * i + j] = 0.0;
		}
		mTV [mYS * i + TVecChar [i]] = 1.0;
	}
	double *mNowX;
	mNowX = new double [mXS];
	for (int i = 0; i < mXS; i++)
		mNowX [i] = XVec [i];
	mNetwork = new ElmanNetwork (n, k, m, mNowX);
	mDeltaH = new double [mHS];
	mDeltaY = new double [mYS];
}

void ElmanTraining::makeDeltas (int nowExample) {
	int i, k;
	double outputK, sumK;
	for (k = 0; k < mYS; k++) {
		outputK = mNetwork -> mYL [k] -> getValue ();
		mDeltaY [k] = - outputK * (1 - outputK) * (mTV [mYS * nowExample + k] - outputK);
	}
	for (k = 0; k < mHS; k++) {
		outputK = mNetwork -> mHL [k] -> getValue ();
		sumK = 0;
		for (i = 0; i < mYS; i++) {
			sumK += ( mNetwork -> mHL [k] -> getWeight(i) ) * mDeltaY [i];
		}
		mDeltaH [k] = - outputK * (1 - outputK) * sumK;
	}
}

void ElmanTraining::changeWeights (double eta) {
	int i, j;
	for (i = 0; i < mXS; i++) {
		for (j = 0; j < mHS; j++) {
			mNetwork -> mXL [i] -> changeWeight(j, -eta * mDeltaH [j]);
		}
	}
	for (i = 0; i < mHS; i++) {
		for (j = 0; j < mHS; j++) {
			mNetwork -> mCL [i] -> changeWeight(j, -eta * mDeltaH [j]);
		}
		for (j = 0; j < mYS; j++) {
			mNetwork -> mHL [i] -> changeWeight(j, -eta * mDeltaY [j]);
		}
	}
}

void ElmanTraining::saveWeigths (char *filename) {
	std::fstream fout;
	fout.open (filename, std::ios::out);
	int i, j;
	for (i = 0; i < mXS; i++) {
		for (j = 0; j < mHS; j++) {
			fout << mNetwork -> mXL [i] -> getWeight(j) << std::endl;
		}
	}
	for (i = 0; i < mHS; i++) {
		for (j = 0; j < mHS; j++) {
			fout << mNetwork -> mCL [i] -> getWeight(j) << std::endl;
		}
	}
	for (i = 0; i < mHS; i++) {
		for (j = 0; j < mYS; j++) {
			fout << mNetwork -> mHL [i] -> getWeight(j) << std::endl;
		}
	}
	fout.close ();
}

void ElmanTraining::Train(int epochN, int iterationLim, double eta) {
	int nowExample, i;
	double *mNowX;
	mNowX = new double [mXS];
	for (int epoch = 0; epoch < epochN; epoch++) {
		for (nowExample = 0; nowExample < mNumExamples; nowExample++) {
			for (i = 0; i < mXS; i++)
				mNowX [i] = mXV [nowExample * mXS + i];
			mNetwork -> setX (mNowX);
			mNetwork -> TrainIterate();
			for (i = 0; i < iterationLim; i++) {
				makeDeltas (nowExample);
				changeWeights (eta);
				mNetwork -> MakeContextLayer();
			}
		}
	}
}
