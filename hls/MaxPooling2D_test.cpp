#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "MaxPooling2D.h"

int main(void)
{
    cv::Mat inMat = cv::imread("/mnt/nvme0n1p5/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat inMat = cv::imread("/home/emanuel/workspace/datasets/desktop_dataset/images/scene_000.png", cv::IMREAD_GRAYSCALE);

    //cv::resize(inMat, inMat, cv::Size(MAX_WIDTH, MAX_WIDTH), cv::INTER_AREA);
    
    int width = inMat.cols;
    int height = inMat.rows;

    std::cout << "height " << height << " width " << width << std::endl;

    cv::Mat outMat(height/2, width/2, CV_32FC1, cv::Scalar(0));

	hls::stream<mp2D_packet> s_in;
	hls::stream<mp2D_packet> s_out;

    for (int y = 0; y < height; y++)
	{
        for(int x = 0; x < width; x++)
        {
            mp2D_packet in_packet;

            in_packet.data = float(inMat.at<uchar>(y, x));
            in_packet.last = false;
            in_packet.keep = -1;
            if (x == width - 1 && y == height - 1)
                in_packet.last = true;
            s_in.write(in_packet);
        }
	}

    MaxPooling2D(s_in, s_out, width, height); 

    for (int y = 0; y < height/2; y++)
	{
        for(int x = 0; x < width/2; x++)
        {
            mp2D_packet out_packet;
            s_out.read(out_packet);

            outMat.at<float>(y, x) = out_packet.data;
        }
	}

    cv::imwrite("MaxPoolingOut.png", outMat);

    return 0;
}
