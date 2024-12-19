#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "conv2D_3x3.h"

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> packet_type;

int main(void)
{
    // cv::Mat inMat = cv::imread("/mnt/nvme0n1p5/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat inMat = cv::imread("/home/emanuel/workspace/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);

    // cv::resize(inMat, inMat, cv::Size(MAX_WIDTH, MAX_WIDTH), cv::INTER_AREA);

    int in_width = inMat.cols;
    int in_height = inMat.rows;

    int padding = 1;

    int out_width = in_width - 2 + padding * 2;
    int out_height = in_height - 2 + padding * 2;

    std::cout << "in_height " << in_height << " in_width " << in_width << std::endl;
    std::cout << "out_height " << out_height << " out_width " << out_width << std::endl;

    cv::Mat outMat(out_height, out_width, CV_32FC1, cv::Scalar(0));
    cv::Mat diffMat(out_height, out_width, CV_32FC1, cv::Scalar(0));

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

    float kernel[3 * 3] = {0};

    /*
        kernel[0] = 1.0;
        kernel[1] = 2.0;
        kernel[2] = 1.0;

        kernel[3] = 0.0;
        kernel[4] = 0.0;
        kernel[5] = 0.0;

        kernel[6] = -1.0;
        kernel[7] = -2.0;
        kernel[8] = -1.0;
    */

    // kernel[4] = 1.0;

    kernel[0] = 0.0;
    kernel[1] = 1.0;
    kernel[2] = 0.0;

    kernel[3] = 1.0;
    kernel[4] = -4.0;
    kernel[5] = 1.0;

    kernel[6] = 0.0;
    kernel[7] = 1.0;
    kernel[8] = 0.0;

    /*
        kernel[0] = 1.0/9.0;
        kernel[1] = 1.0/9.0;
        kernel[2] = 1.0/9.0;

        kernel[3] = 1.0/9.0;
        kernel[4] = 1.0/9.0;
        kernel[5] = 1.0/9.0;

        kernel[6] = 1.0/9.0;
        kernel[7] = 1.0/9.0;
        kernel[8] = 1.0/9.0;
    */
    conv2D_3x3<float, packet_type, 480, 640, 1>(s_in, s_out, kernel);

    for (int y = 0; y < out_height; y++)
    {
        for (int x = 0; x < out_width; x++)
        {
            packet_type out_packet;
            s_out.read(out_packet);
            float data = out_packet.data;

            outMat.at<float>(y, x) = data;
            diffMat.at<float>(y, x) = fabs(data - float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding)));
            // diffMat.at<float>(y, x) = float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding));
        }
    }

    cv::imwrite("conv2D_3x3_filtered.png", outMat);
    cv::imwrite("conv2D_3x3_diff.png", diffMat);

    return 0;
}
