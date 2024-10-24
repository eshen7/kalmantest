#include "Kalman.h"

#include <iostream>

Kalman::Kalman() = default;

void Kalman::init(double initTime, Eigen::Matrix<double, 4, 1> &initEstimate) {
    time = initTime;
    isInit = true;
    currentStateEstimate.timestamp = initTime;
    currentStateEstimate.data = initEstimate;
    currentStatePrediction.timestamp = initTime;
    currentStatePrediction.data = initEstimate;
    priorStateEstimate = initEstimate;
    currentMeasurementMatrix.timestamp = initTime;
    currentMeasurementMatrix.data = initEstimate;
}

void Kalman::updateTime(double newTime) {
    time = newTime;
}

void Kalman::updateTransitionMatrix() {
    // call after updating time but before updating predictions
    double deltaTime = time - currentStateEstimate.timestamp;
    stateTransitionMatrix(0, 2) = deltaTime;
    stateTransitionMatrix(1, 3) = deltaTime;
}


void Kalman::predict() {
    currentStatePrediction.timestamp = time;
    currentStatePrediction.data = stateTransitionMatrix * priorStateEstimate;
}

void Kalman::updateMeasurement(int x, int y) {
    double deltaTime = time - currentStateEstimate.timestamp;
    currentMeasurementMatrix.timestamp = time;
    currentMeasurementMatrix.data(0, 0) = x;
    currentMeasurementMatrix.data(1, 0) = y;
    currentMeasurementMatrix.data(2, 0) = (x - priorStateEstimate(0, 0)) / deltaTime;
    currentMeasurementMatrix.data(3, 0) = (y - priorStateEstimate(1, 0)) / deltaTime;
}


void Kalman::stateUpdate(int x, int y) {
    updateMeasurement(x, y);
    priorStateEstimate = currentStateEstimate.data;
    currentStateEstimate.timestamp = time;
    currentStateEstimate.data = currentStatePrediction.data + currentKalmanGain * (
                                    currentMeasurementMatrix.data - currentStatePrediction.data);
}

int Kalman::getTime() {
    return time;
}

std::tuple<int, int> Kalman::trajPred(double deltaTime) {
    futurePredMatrix(0,2) = deltaTime;
    futurePredMatrix(1,3) = deltaTime;
    Eigen::MatrixXd futureMat = futurePredMatrix * currentStateEstimate.data;
    return {futureMat(0, 0), futureMat(1, 0)};
}

void Kalman::update(double newTime, int x, int y) {
    updateTime(newTime);
    updateTransitionMatrix();
    predict();
    stateUpdate(x, y);
}

bool Kalman::getIsInit() {
    return isInit;
}
