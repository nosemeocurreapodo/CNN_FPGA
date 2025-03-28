#pragma once

#include "conv2D_3x3_IM_params.h"

const w_data_type kernel_identity[3 * 3] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
const w_data_type kernel_sobel_py[3 * 3] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};
const w_data_type kernel_sobel_ny[3 * 3] = {-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0};
const w_data_type kernel_sobel_px[3 * 3] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};
const w_data_type kernel_sobel_nx[3 * 3] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
const w_data_type kernel_laplacian[3 * 3] = {0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0};

inline void conv2D_3x3_IM_weights(mat3<w_data_type> kernel[in_channels][out_channels])
{
    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            int index = in_channel * out_channels + out_channel;
            mat3<w_data_type> mat_touse; //(kernel_identity);

            if (index % 6 == 0)
                mat_touse = mat3<w_data_type>(kernel_identity);
            if (index % 6 == 1)
                mat_touse = mat3<w_data_type>(kernel_sobel_py);
            if (index % 6 == 2)
                mat_touse = mat3<w_data_type>(kernel_sobel_px);
            if (index % 6 == 3)
                mat_touse = mat3<w_data_type>(kernel_sobel_ny);
            if (index % 6 == 4)
                mat_touse = mat3<w_data_type>(kernel_sobel_nx);
            if (index % 6 == 5)
                mat_touse = mat3<w_data_type>(kernel_laplacian);

            kernel[in_channel][out_channel] = mat_touse;
        }
    }
}

inline void conv2D_3x3_random_weights(w_data_type kernel[in_channels * out_channels * 3 * 3])
{
    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    int index = in_channel * out_channels * 3 * 3 + out_channel * 3 * 3 + y * 3 + x;

                    kernel[index] = w_data_type(float(index) / 10.0);
                }
            }
        }
    }
}

inline void conv2D_3x3_random_weights(mat3<w_data_type> kernel[in_channels][out_channels])
{
    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    int index = in_channel * out_channels + out_channel * 3 * 3 + y * 3 + x;
                    kernel[in_channel][out_channel].data[y][x] = w_data_type(float(index) / 10.0);
                }
            }
        }
    }
}