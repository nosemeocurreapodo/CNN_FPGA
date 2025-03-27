#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "MaxPooling2D.h"

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> packet_type;

int main(void)
{
    //cv::Mat inMat = cv::imread("/mnt/nvme0n1p5/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat inMat = cv::imread("/home/emanuel/workspace/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);

    // cv::resize(inMat, inMat, cv::Size(MAX_WIDTH, MAX_WIDTH), cv::INTER_AREA);

    int in_width = inMat.cols;
    int in_height = inMat.rows;

    int out_width = in_width/2;
    int out_height = in_height/2;

    std::cout << "in_height " << in_height << " in_width " << in_width << std::endl;
    std::cout << "out_height " << out_height << " out_width " << out_width << std::endl;

    cv::Mat outMat(out_height, out_width, CV_32FC1, cv::Scalar(0));

    hls::stream<packet_type> s_in;
    hls::stream<packet_type> s_out;

    for (int y = 0; y < in_height; y++)
    {
        for (int x = 0; x < in_width; x++)
        {
            packet_type in_packet;

            in_packet.data = float(inMat.at<uchar>(y, x));
            in_packet.last = false;
            in_packet.keep = -1;
            if (x == in_width - 1 && y == in_height - 1)
                in_packet.last = true;
            s_in.write(in_packet);
        }
    }

    MaxPooling2D<float, packet_type, 480, 640>(s_in, s_out);

    for (int y = 0; y < out_height; y++)
    {
        for (int x = 0; x < out_width; x++)
        {
            packet_type out_packet;
            s_out.read(out_packet);
            outMat.at<float>(y, x) = out_packet.data;
        }
    }

    cv::imwrite("MaxPoolingOut.png", outMat);

    return 0;
}
