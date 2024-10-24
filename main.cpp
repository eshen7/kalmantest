#include "Kalman.cpp"
#include <opencv2/opencv.hpp>
#include <optional>

std::optional<std::tuple<int, int> > detectRedObj(cv::Mat &frame) {
    cv::Mat hsvImage;
    cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);

    // Define the range of red color in HSV
    cv::Scalar lower_pink(145, 100, 100); // Lower bound of pink
    cv::Scalar upper_pink(179, 255, 255); // Upper bound of pink

    // Threshold the HSV image to get only pink colors
    cv::Mat mask;
    cv::inRange(hsvImage, lower_pink, upper_pink, mask);

    // Find contours in the mask
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour (if any)
    if (!contours.empty()) {
        std::vector<cv::Point> largestContour = contours[0];
        double maxArea = cv::contourArea(largestContour);

        for (const auto &contour: contours) {
            double area = cv::contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        // Calculate the center of the largest contour
        cv::Moments m = cv::moments(largestContour);
        if (m.m00 > 0) {
            int centerX = static_cast<int>(m.m10 / m.m00);
            int centerY = static_cast<int>(m.m01 / m.m00);

            // Draw a circle at the center
            cv::circle(frame, cv::Point(centerX, centerY), 20, cv::Scalar(0, 255, 0), -1); // Green dot
            return std::tuple<int, int>{centerX, centerY};
        }
    } else {
        return std::nullopt;
    }
}

int main() {
    Eigen::Matrix<double, 4, 1> init;
    Kalman K;
    int deviceID = 0;
    cv::VideoCapture cap(deviceID);
    cv::namedWindow("test");
    if (!cap.isOpened()) {
        std::cerr << "could not open camera\n";
        return 0;
    }
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "no image detected";
            return 0;
        }
        std::optional<std::tuple<int, int> > detectedCoordinates = detectRedObj(frame);
        if (detectedCoordinates.has_value()) {
            auto [x, y] = detectedCoordinates.value();
            if (!K.getIsInit()) {
                init << x, y, 0, 0;
                K.init(std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start).count(), init);
            }
            else {
                K.update(std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::high_resolution_clock::now() - start).count(), x, y);
                auto [futureX, futureY] = K.trajPred(10);
                cv::circle(frame, cv::Point(futureX, futureY), 20, cv::Scalar(255, 0, 0), -1); // Green dot
            }
        }
        cv::imshow("test", frame);
        cv::waitKey(20);
    }
    return 0;
}
