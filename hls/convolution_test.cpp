#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>

#include "convolution.h"

int main(void)
{
    int width = 480;
    int height = 480;

    cv::Mat imageMat = cv::imread("/test_data/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::resize(imageMat, imageMat, cv::Size(width, height), cv::INTER_AREA);

    cv::imwrite("/test_data/scene_000_filtered.png", imageMat);

    return 1;
}
