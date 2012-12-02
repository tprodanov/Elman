#pragma once

class Neuron
{
public:
	Neuron();
	Neuron(int WeightsVectorSize, double NeuronValue);
	Neuron(int WeightsVectorSize);
	double getValue (int i);
	double getValue ();
	void setValue (double NeuronValue);
	void setWeight (int, double);
	double getWeight (int);
	void changeWeight (int, double);
private:
	int mSize;
	double mValue;
	double *mW;
};
