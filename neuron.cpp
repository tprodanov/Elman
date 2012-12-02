#include <stdio.h>
#include <stdlib.h>

#include <iostream> // <-delete

#include "neuron.h"

Neuron::Neuron (){}

Neuron::Neuron(int size, double value)
{
	mSize = size;
	mValue = value;
	mW = new double [mSize];
	for (int i = 0; i < mSize; i++) {
		mW [i] = (double) rand () / (double) RAND_MAX;
	}
}

Neuron::Neuron (int size){
	mSize = size;
	mW = new double [mSize];
	for (int i = 0; i < mSize; i++) {
		mW [i] = (double) rand () / (double) RAND_MAX;
	}
}

double Neuron::getValue (int i) {
	return mValue * mW [i];
}

void Neuron::setValue(double value) {
	mValue = value;
}

double Neuron::getValue() {
	return mValue;
}

void Neuron::setWeight (int dest, double weight) {
	mW [dest] = weight;
}

double Neuron::getWeight (int dest) {
	return mW [dest];
}

void Neuron::changeWeight (int dest, double delta) {
	mW [dest] += delta * mValue;
}
