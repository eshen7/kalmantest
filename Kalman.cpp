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
    priorStateEstimate.data = initEstimate;
    priorStateEstimate.timestamp = initTime;
    currentMeasurementMatrix.timestamp = initTime;
    currentMeasurementMatrix.data = initEstimate;
    currentKalmanGain(3, 3) = 0.2;
    currentKalmanGain(2, 2) = 0.2;
    currentKalmanGain(1, 1) = 0.8;
    currentKalmanGain(0, 0) = 0.8;
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
    currentStatePrediction.data = stateTransitionMatrix * priorStateEstimate.data;
}

void Kalman::updateMeasurementPosition(int x, int y) {
    double deltaTime = time - currentStateEstimate.timestamp;
    currentMeasurementMatrix.timestamp = time;
    currentMeasurementMatrix.data(0, 0) = x;
    currentMeasurementMatrix.data(1, 0) = y;
}

void Kalman::updateMeasurementVelocity() {
    double deltaTime = currentStateEstimate.timestamp - priorStateEstimate.timestamp;
    std::cout << (currentStateEstimate.data(0, 0) - priorStateEstimate.data(0, 0)) / deltaTime << std::endl;
    currentMeasurementMatrix.data(2, 0) = (currentStateEstimate.data(0, 0) - priorStateEstimate.data(0, 0)) / deltaTime;
    currentMeasurementMatrix.data(3, 0) = (currentStateEstimate.data(1, 0) - priorStateEstimate.data(1, 0)) / deltaTime;
}


void Kalman::stateUpdate(int x, int y) {
    updateMeasurementPosition(x, y);
    priorStateEstimate.data = currentStateEstimate.data;
    priorStateEstimate.timestamp = currentStateEstimate.timestamp;
    currentStateEstimate.timestamp = time;
    currentStateEstimate.data = currentStatePrediction.data + currentKalmanGain * (
                                    currentMeasurementMatrix.data - currentStatePrediction.data);
    updateMeasurementVelocity();
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
