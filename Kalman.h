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
    void updateMeasurement(int x, int y);
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
    Eigen::Matrix<double, 4, 1> priorStateEstimate;
    Eigen::Matrix<double, 4, 4> futurePredMatrix = Eigen::MatrixXd::Identity(4, 4);
    Eigen::Matrix<double, 4, 4> stateTransitionMatrix = Eigen::MatrixXd::Identity(4, 4);
    Eigen::Matrix<double, 4, 4> currentKalmanGain = Eigen::MatrixXd::Identity(4, 4) * 0.5; // constant for testing;
};

#endif