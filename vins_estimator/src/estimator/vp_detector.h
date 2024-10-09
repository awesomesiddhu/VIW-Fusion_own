#include <ros/ros.h>
#include <stdio.h>
#include <iostream>
#include "std_msgs/String.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <armadillo>
#include <opencv2/ximgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
std::vector<double> vp_detector(cv::Mat img);
using namespace std;
using namespace cv;