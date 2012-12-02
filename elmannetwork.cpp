#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

#include <iostream>

#include "elmannetwork.h"

ElmanNetwork::ElmanNetwork(int n, int k, int m, double *vec)
{
	mXS = n;
	mHS = k;
	mYS = m;
	srand ( time(NULL) );
	mXL = new Neuron *[mXS];
	int i;
	for (i = 0; i < mXS; i++) {
		mXL [i] = new Neuron (mHS, vec [i]);
	}
	mCL = new Neuron *[mHS];
	for (i = 0; i < mHS; i++) {
		mCL [i] = new Neuron (mHS, 0.5);
	}
	mHL = new Neuron *[mHS];
	for (i = 0; i < mHS; i++) {
		mHL [i] = new Neuron (mYS);
	}
	mYL = new Neuron *[mYS];
	for (i = 0; i < mYS; i++) {
		mYL [i] = new Neuron (0);
	}
}

ElmanNetwork::ElmanNetwork(int n, int k, int m, double *vec, char *filename)
{
	mXS = n;
	mHS = k;
	mYS = m;
	srand ( time(NULL) );
	mXL = new Neuron *[mXS];
	int i;
	for (i = 0; i < mXS; i++) {
		mXL [i] = new Neuron (mHS, vec [i]);
	}
	mCL = new Neuron *[mHS];
	for (i = 0; i < mHS; i++) {
		mCL [i] = new Neuron (mHS, 0.5);
	}
	mHL = new Neuron *[mHS];
	for (i = 0; i < mHS; i++) {
		mHL [i] = new Neuron (mYS);
	}
	mYL = new Neuron *[mYS];
	for (i = 0; i < mYS; i++) {
		mYL [i] = new Neuron (0);
	}
	getWeights (filename);
}

void ElmanNetwork::getWeights (char *filename) {
	std::fstream fin;
	fin.open (filename, std::ios::in);
	int i, j;
	double tmp;
	for (i = 0; i < mXS; i++) {
		for (j = 0; j < mHS; j++) {
			fin >> tmp;
			mXL [i] -> setWeight(j, tmp);
		}
	}
	for (i = 0; i < mHS; i++) {
		for (j = 0; j < mHS; j++) {
			fin >> tmp;
			mCL [i] -> setWeight(j, tmp);
		}
	}
	for (i = 0; i < mHS; i++) {
		for (j = 0; j < mYS; j++) {
			fin >> tmp;
			mHL [i] -> setWeight(j, tmp);
		}
	}
	fin.close ();
}

void ElmanNetwork::setX (double *vec) {
	int i;
	for (i = 0; i < mXS; i++) {
		mXL [i] -> setValue (vec [i]);
	}
	for (i = 0; i < mHS; i++) {
		mCL [i] -> setValue (0.5);
	}
}

double ElmanNetwork::mf1 (double x) {
	return 1/(1 + exp (-2 * mAlphaF * x));
}

double ElmanNetwork::mf2 (double x) {
	return 1/(1 + exp (-2 * mAlphaF * x));
}

double ElmanNetwork::CalculateHNeuron (int i) {
	double res = 0;
	for (int j = 0; j < mXS; j++) {
		res += mXL [j] -> getValue (i);
	}
	for (int j = 0; j < mHS; j++) {
		res += mCL [j] -> getValue (i);
	}
	return mf1 (res);
}

double ElmanNetwork::CalculateYNeuron (int i) {
	double res = 0;
	for (int j = 0; j < mHS; j++) {
		res += mHL [j] -> getValue (i);
	}
	return mf2 (res);
}

void ElmanNetwork::Iterate () {
	int i;
	for (i = 0; i < mHS; i++) {
		mCL [i] -> setValue (mHL [i] -> getValue ());
	}
	for (i = 0; i < mHS; i++) {
		mHL [i] -> setValue (CalculateHNeuron (i));
	}
}

void ElmanNetwork::TrainIterate () {
	int i;
	for (i = 0; i < mHS; i++) {
		mHL [i] -> setValue (CalculateHNeuron (i));
	}
	for (i = 0; i < mYS; i++) {
		mYL [i] -> setValue (CalculateYNeuron (i));
	}
}

void ElmanNetwork::MakeContextLayer () {
	for (int i = 0; i < mHS; i++) {
		mCL [i] -> setValue (mHL [i] -> getValue ());
	}
}

void ElmanNetwork::Recognize (int iterationLimit) {
	int i;
	for (i = 0; i < mHS; i++) {
		mHL [i] -> setValue(CalculateHNeuron (i));
	}
	int iteration = 1;
	while (iteration < iterationLimit) {
		Iterate ();
		iteration++;
	}
	for (i = 0; i < mYS; i++) {
		mYL [i] -> setValue (CalculateYNeuron (i));
		std::cout << mYL [i] ->getValue() << " ";
	}
}
