#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "conv2D_Nx3x3.h"

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> packet_type;

int main(void)
{
    // cv::Mat inMat = cv::imread("/mnt/nvme0n1p5/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat inMat = cv::imread("/home/emanuel/workspace/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);

    // cv::resize(inMat, inMat, cv::Size(MAX_WIDTH, MAX_WIDTH), cv::INTER_AREA);

    int in_width = inMat.cols;
    int in_height = inMat.rows;
    int in_channels = 3;

    int padding = 1;

    int out_width = in_width - 2 + padding * 2;
    int out_height = in_height - 2 + padding * 2;
    int out_channels = in_channels;

    std::cout << "in_height " << in_height << " in_width " << in_width << std::endl;
    std::cout << "out_height " << out_height << " out_width " << out_width << std::endl;

    cv::Mat outMat(out_height, out_width, CV_32FC1, cv::Scalar(0));
    cv::Mat diffMat(out_height, out_width, CV_32FC1, cv::Scalar(0));

    hls::stream<packet_type> s_in;
    hls::stream<packet_type> s_out;
    hls::stream<packet_type> s_kernel;

    float kernel[3 * 3 * 3] = {0};
    // identity
    kernel[4] = 1.0;
    // sobel
    kernel[0 + 9] = 1.0;
    kernel[1 + 9] = 2.0;
    kernel[2 + 9] = 1.0;

    kernel[3 + 9] = 0.0;
    kernel[4 + 9] = 0.0;
    kernel[5 + 9] = 0.0;

    kernel[6 + 9] = -1.0;
    kernel[7 + 9] = -2.0;
    kernel[8 + 9] = -1.0;
    // laplacian
    kernel[0 + 18] = 0.0;
    kernel[1 + 18] = 1.0;
    kernel[2 + 18] = 0.0;

    kernel[3 + 18] = 1.0;
    kernel[4 + 18] = -4.0;
    kernel[5 + 18] = 1.0;

    kernel[6 + 18] = 0.0;
    kernel[7 + 18] = 1.0;
    kernel[8 + 18] = 0.0;

    for (int channel = 0; channel < in_channels; channel++)
    {
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                packet_type in_packet;

                in_packet.data = kernel[x + y * 3 + channel * 9];
                in_packet.last = false;
                in_packet.keep = -1;
                if (x == 3 - 1 && y == 3 - 1 && channel == in_channels - 1)
                    in_packet.last = true;
                s_kernel.write(in_packet);
            }
        }
    }

    for (int channel = 0; channel < in_channels; channel++)
    {
        for (int y = 0; y < in_height; y++)
        {
            for (int x = 0; x < in_width; x++)
            {
                packet_type in_packet;

                in_packet.data = float(inMat.at<uchar>(y, x));
                in_packet.last = false;
                in_packet.keep = -1;
                if (x == in_width - 1 &&y == in_height - 1 && channel == in_channels - 1)
                    in_packet.last = true;
                s_in.write(in_packet);
            }
        }
    }

    conv2D_Nx3x3<float, packet_type, 3, 480, 640, 1>(s_in, s_out, s_kernel);

    for (int channel = 0; channel < in_channels; channel++)
    {
        for (int y = 0; y < out_height; y++)
        {
            for (int x = 0; x < out_width; x++)
            {
                packet_type out_packet;
                s_out.read(out_packet);
                float data = out_packet.data;

                if(channel == 1)
                {
                    outMat.at<float>(y, x) = data;
                    diffMat.at<float>(y, x) = fabs(data - float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding)));
                    // diffMat.at<float>(y, x) = float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding));
                }
            }
        }
    }

    cv::imwrite("conv2D_Nx3x3_filtered.png", outMat);
    cv::imwrite("conv2D_Nx3x3_diff.png", diffMat);

    return 0;
}
