cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CXX_STANDARD 17)

project(kalmantest)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION bin)