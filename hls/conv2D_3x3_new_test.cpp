#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "conv2D_3x3_new.h"

#define IN_CHANNELS 1
#define OUT_CHANNELS 3
#define PADDING 1

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> packet_type;

int main(void)
{
    // cv::Mat inMat = cv::imread("/mnt/nvme0n1p5/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat inMat = cv::imread("/home/emanuel/workspace/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);

    // cv::resize(inMat, inMat, cv::Size(MAX_WIDTH, MAX_WIDTH), cv::INTER_AREA);

    int in_width = inMat.cols;
    int in_height = inMat.rows;

    int out_width = in_width - 2 + PADDING * 2;
    int out_height = in_height - 2 + PADDING * 2;

    std::cout << "in_height " << in_height << " in_width " << in_width << std::endl;
    std::cout << "out_height " << out_height << " out_width " << out_width << std::endl;

    cv::Mat outMat[OUT_CHANNELS];
    cv::Mat diffMat[OUT_CHANNELS];

    for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++)
    {
        outMat[out_channel] = cv::Mat(out_height, out_width, CV_32FC1, cv::Scalar(0));
        diffMat[out_channel] = cv::Mat(out_height, out_width, CV_32FC1, cv::Scalar(0));
    }

    hls::stream<packet_type> s_in;
    hls::stream<packet_type> s_out;
    hls::stream<packet_type> s_kernel;

    float kernel[IN_CHANNELS * OUT_CHANNELS * 3 * 3];
    // identity
    kernel[0] = 0.0;
    kernel[1] = 0.0;
    kernel[2] = 0.0;

    kernel[3] = 0.0;
    kernel[4] = 1.0;
    kernel[5] = 0.0;

    kernel[6] = 0.0;
    kernel[7] = 0.0;
    kernel[8] = 0.0;

    // sobel
    kernel[0 + 9] = 0.0;//1.0;
    kernel[1 + 9] = 0.0;//2.0;
    kernel[2 + 9] = 0.0;//1.0;

    kernel[3 + 9] = 0.0;
    kernel[4 + 9] = 1.0;//0.0;
    kernel[5 + 9] = 0.0;

    kernel[6 + 9] = 0.0;//-1.0;
    kernel[7 + 9] = 0.0;//-2.0;
    kernel[8 + 9] = 0.0;//-1.0;

    // laplacian
    kernel[0 + 18] = 0.0;
    kernel[1 + 18] = 0.0;//1.0;
    kernel[2 + 18] = 0.0;

    kernel[3 + 18] = 0.0;//1.0;
    kernel[4 + 18] = 1.0;//-4.0;
    kernel[5 + 18] = 0.0;//1.0;

    kernel[6 + 18] = 0.0;
    kernel[7 + 18] = 0.0;//1.0;
    kernel[8 + 18] = 0.0;

    for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++)
    {
        for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++)
        {
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    packet_type in_packet;

                    in_packet.data = kernel[x + y * 3 + out_channel * 9 + in_channel * OUT_CHANNELS * 9];
                    in_packet.last = false;
                    in_packet.keep = -1;
                    if (x == 3 - 1 && y == 3 - 1 && in_channel == IN_CHANNELS - 1 && out_channel == OUT_CHANNELS - 1)
                        in_packet.last = true;
                    s_kernel.write(in_packet);
                }
            }
        }
    }

    for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++)
    {
        for (int y = 0; y < in_height; y++)
        {
            for (int x = 0; x < in_width; x++)
            {
                packet_type in_packet;

                in_packet.data = float(inMat.at<uchar>(y, x));
                in_packet.last = false;
                in_packet.keep = -1;
                if (x == in_width - 1 && y == in_height - 1 && in_channel == IN_CHANNELS - 1)
                    in_packet.last = true;
                s_in.write(in_packet);
            }
        }
    }

    conv2D_3x3<float, packet_type, IN_CHANNELS, OUT_CHANNELS, 480, 640, PADDING>(s_in, s_out, s_kernel);

    for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++)
    {
        for (int y = 0; y < out_height; y++)
        {
            for (int x = 0; x < out_width; x++)
            {
                packet_type out_packet;
                s_out.read(out_packet);
                float data = out_packet.data/IN_CHANNELS;

                outMat[out_channel].at<float>(y, x) = data;
                diffMat[out_channel].at<float>(y, x) = fabs(data - float(inMat.at<uchar>(y + 1 - PADDING, x + 1 - PADDING)));
                // diffMat.at<float>(y, x) = float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding));
            }
        }
    }

    cv::imwrite("conv2D_3x3_new_filtered.png", outMat[0]);
    cv::imwrite("conv2D_3x3_new_diff.png", diffMat[0]);

    return 0;
}
