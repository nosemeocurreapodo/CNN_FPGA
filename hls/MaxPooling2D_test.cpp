#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "MaxPooling2D.h"

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> packet_type;

int main(void)
{
    cv::Mat inMat = cv::imread("/mnt/nvme0n1p5/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat inMat = cv::imread("/home/emanuel/workspace/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);

    // cv::resize(inMat, inMat, cv::Size(MAX_WIDTH, MAX_WIDTH), cv::INTER_AREA);

    int width = inMat.cols;
    int height = inMat.rows;

    std::cout << "height " << height << " width " << width << std::endl;

    cv::Mat outMat(height / 2, width / 2, CV_32FC1, cv::Scalar(0));

    hls::stream<packet_type> s_in;
    hls::stream<packet_type> s_out;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            packet_type in_packet;

            in_packet.data = float(inMat.at<uchar>(y, x));
            in_packet.last = false;
            in_packet.keep = -1;
            if (x == width - 1 && y == height - 1)
                in_packet.last = true;
            s_in.write(in_packet);
        }
    }

    MaxPooling2D<float, packet_type, 640, 480>(s_in, s_out);

    for (int y = 0; y < height / 2; y++)
    {
        for (int x = 0; x < width / 2; x++)
        {
            packet_type out_packet;
            s_out.read(out_packet);
            outMat.at<float>(y, x) = out_packet.data;
        }
    }

    cv::imwrite("MaxPoolingOut.png", outMat);

    return 0;
}
