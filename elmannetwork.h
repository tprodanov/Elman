#pragma once

#include "neuron.h"
#include "elmantraining.h"

class ElmanNetwork
{
	friend class ElmanTraining;
public:
	ElmanNetwork(int, int, int, double*);
	ElmanNetwork(int, int, int, double*, char*);
	void setX (double*);
	void Recognize (int);
	void getWeights (char*);
private:
	double mf1 (double);
	double mf2 (double);
	double CalculateHNeuron (int);
	double CalculateYNeuron (int);
	void Iterate ();
	void TrainIterate ();
	void MakeContextLayer ();

	int mXS, mHS, mYS;
	//X Layer size, Hidden Layer size, Y Layer size
	Neuron **mXL, **mCL, **mHL, **mYL;
	// X-Layer, Context-Layer, Hidden-Layer, Y-Layer
	static const double mAlphaF = 0.7;
};
