cmake_minimum_required(VERSION 3.24)

project(opencv_viewer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio features2d)

qt_add_executable(opencv_viewer
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
)

target_link_libraries(opencv_viewer
    PRIVATE
        Qt6::Widgets
        ${OpenCV_LIBS}
)

target_include_directories(opencv_viewer
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

set_target_properties(opencv_viewer PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
)
