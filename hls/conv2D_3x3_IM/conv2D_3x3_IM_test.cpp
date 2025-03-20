#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "conv2D_3x3_IM.h"
#include "conv2D_3x3_IM_params.h"

int main(int argc, char *argv[])
{
    // std::cout << argv[0] << std::endl;
    std::string input_folder = "/home/emanuel/workspace/CNN_FPGA/hls/test_data";
    std::string output_folder = "/home/emanuel/workspace/CNN_FPGA/hls/results";
    std::string input_file_path = input_folder + "/scene_000.png";
    cv::Mat inMat = cv::imread(input_file_path, cv::IMREAD_GRAYSCALE);

    // cv::resize(inMat, inMat, cv::Size(MAX_WIDTH, MAX_WIDTH), cv::INTER_AREA);

    int in_width = inMat.cols;
    int in_height = inMat.rows;

    int out_width = in_width - 2 + padding * 2;
    int out_height = in_height - 2 + padding * 2;

    std::cout << "in_height " << in_height << " in_width " << in_width << std::endl;
    std::cout << "out_height " << out_height << " out_width " << out_width << std::endl;

    cv::Mat outMat[in_channels][out_channels];
    cv::Mat diffMat[in_channels][out_channels];

    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            outMat[in_channel][out_channel] = cv::Mat(out_height, out_width, CV_32FC1, cv::Scalar(0));
            diffMat[in_channel][out_channel] = cv::Mat(out_height, out_width, CV_32FC1, cv::Scalar(0));
        }
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

    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int y = 0; y < out_height; y++)
        {
            for (int x = 0; x < out_width; x++)
            {
                for (int out_channel = 0; out_channel < out_channels; out_channel++)
                {
                    packet_type out_packet;
                    s_out.read(out_packet);
                    float data = out_packet.data;

                    outMat[in_channel][out_channel].at<float>(y, x) = data;
                    diffMat[in_channel][out_channel].at<float>(y, x) = fabs(data - float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding)));
                    // diffMat.at<float>(y, x) = float(inMat.at<uchar>(y + 1 - padding, x + 1 - padding));
                }
            }
        }
    }

    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            std::string filtered_file_name = output_folder + "/conv2D_3x3_IM_filtered_" + std::to_string(in_channel) + "_" + std::to_string(out_channel) + ".png";
            std::string diff_file_name = output_folder + "/conv2D_3x3_IM_diff_" + std::to_string(in_channel) + "_" + std::to_string(out_channel) + ".png";

            cv::imwrite(filtered_file_name, outMat[in_channel][out_channel]);
            cv::imwrite(diff_file_name, diffMat[in_channel][out_channel]);
        }
    }

    return 0;
}
