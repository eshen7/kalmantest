#ifndef KALMAN_H
#define KALMAN_H

#include <Eigen/Eigen>

struct timestampedMatrix {
    double timestamp;
    Eigen::Matrix<double, 4, 1> data;
};

class Kalman {
    public:
    Kalman();
    void init(double initTime, Eigen::Matrix<double, 4, 1>& initEstimate);
    void updateTime(double newTime);
    void updateTransitionMatrix();
    void update(double newTime, int x, int y);
    void updateMeasurementPosition(int x, int y);
    void updateMeasurementVelocity();
    void stateUpdate(int x, int y);
    void predict();
    std::tuple<int, int> trajPred(double deltaTime);
    int getTime();
    bool getIsInit();
private:
    double time = 0;
    bool isInit = false;
    timestampedMatrix currentStateEstimate = timestampedMatrix(); // x, y, xvelo, yvelo
    timestampedMatrix currentStatePrediction = timestampedMatrix();
    timestampedMatrix currentMeasurementMatrix = timestampedMatrix();
    timestampedMatrix priorStateEstimate = timestampedMatrix();
    Eigen::Matrix<double, 4, 4> futurePredMatrix = Eigen::MatrixXd::Identity(4, 4);
    Eigen::Matrix<double, 4, 4> stateTransitionMatrix = Eigen::MatrixXd::Identity(4, 4);
    Eigen::Matrix<double, 4, 4> currentKalmanGain = Eigen::MatrixXd::Identity(4, 4); // constant for testing;
};

#endif