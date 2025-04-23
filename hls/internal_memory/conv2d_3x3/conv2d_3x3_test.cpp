#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "conv2d_3x3.h"
#include "conv2d_3x3_params.h"

int main(int argc, char *argv[])
{
    // std::cout << argv[0] << std::endl;
    std::string input_folder = "/home/emanuel/workspace/CNN_FPGA/hls/test_data";
    std::string output_folder = "/home/emanuel/workspace/CNN_FPGA/hls/results";
    std::string input_file_path = input_folder + "/scene_000.png";
    cv::Mat inMat = cv::imread(input_file_path, cv::IMREAD_GRAYSCALE);

    cv::Rect box(0, 0, in_width, in_height);
    inMat = inMat(box);

    // This does not compile in Vitis, there is some issue with the libraries
    // cv::resize(inMat, inMat, cv::Size(width, height), cv::INTER_AREA);

    cv::Mat outMat[out_channels];
    cv::Mat diffMat[out_channels];

    const int out_width = in_width - 2 + padding * 2;
    const int out_height = in_height - 2 + padding * 2;

    for (int out_channel = 0; out_channel < out_channels; out_channel++)
    {
        outMat[out_channel] = cv::Mat(out_height, out_width, CV_32FC1, cv::Scalar(0));
        diffMat[out_channel] = cv::Mat(out_height, out_width, CV_32FC1, cv::Scalar(0));
    }

    hls::stream<packet_type> s_in;
    hls::stream<packet_type> s_out;

    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int y = 0; y < in_height; y++)
        {
            for (int x = 0; x < in_width; x++)
            {
                packet_type in_packet;

                in_packet.data = float(inMat.at<uchar>(y, x));
                in_packet.last = false;
                in_packet.keep = -1;
                if (x == in_width - 1 && y == in_height - 1 && in_channel == in_channels - 1)
                    in_packet.last = true;
                s_in.write(in_packet);
            }
        }
    }

    TOP_NAME(s_in, s_out);

    for (int out_channel = 0; out_channel < out_channels; out_channel++)
    {
        for (int y = 0; y < out_height; y++)
        {
            for (int x = 0; x < out_width; x++)
            {
                packet_type out_packet;
                s_out.read(out_packet);
                float data = out_packet.data;

                outMat[out_channel].at<float>(y, x) = data;
                diffMat[out_channel].at<float>(y, x) = fabs(data - float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding)));
                // diffMat.at<float>(y, x) = float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding));
            }
        }
    }

    for (int out_channel = 0; out_channel < out_channels; out_channel++)
    {
        std::string filtered_file_name = output_folder + "/conv2D_3x3_IM_filtered_" + std::to_string(out_channel) + ".png";
        std::string diff_file_name = output_folder + "/conv2D_3x3_IM_diff_" + std::to_string(out_channel) + ".png";

        cv::imwrite(filtered_file_name, outMat[out_channel]);
        cv::imwrite(diff_file_name, diffMat[out_channel]);
    }

    return 0;
}
