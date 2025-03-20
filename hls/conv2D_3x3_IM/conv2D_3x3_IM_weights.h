#pragma once

#include "conv2D_3x3_IM_params.h"

const data_type kernel_identity[3 * 3] = { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
const data_type kernel_sobel_py[3 * 3] = { 1.0, 2.0, 1.0, 0.0, 0.0, 0.0,-1.0,-2.0,-1.0};
const data_type kernel_sobel_ny[3 * 3] = {-1.0,-2.0,-1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0};
const data_type kernel_sobel_px[3 * 3] = { 1.0, 0.0,-1.0, 2.0, 0.0,-2.0, 1.0, 0.0,-1.0};
const data_type kernel_sobel_nx[3 * 3] = {-1.0, 0.0, 1.0,-2.0, 0.0, 2.0,-1.0, 0.0, 1.0};
const data_type kernel_laplacian[3 * 3] = {0.0, 1.0, 0.0, 1.0,-4.0, 1.0, 0.0, 1.0, 0.0};

inline void copy_3x3_kernel(const data_type src[3 * 3], data_type dst[3 * 3])
{
    for (int y = 0; y < 3; y++)
    {
        for (int x = 0; x < 3; x++)
        {
            dst[y * 3 + x] = src[y * 3 + 3];
        }
    }
}

inline void conv2D_3x3_IM_weights(data_type kernel[in_channels * out_channels * 3 * 3])
{
    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
        for (int out_channel = 0; out_channel < out_channels; out_channel++)
        {
            int index = in_channel * out_channels + out_channel;
            data_type kernel_touse[3 * 3];
            if (index % 6 == 0)
                copy_3x3_kernel(kernel_identity, kernel_touse);
            if (index % 6 == 1)
                copy_3x3_kernel(kernel_sobel_py, kernel_touse);
            if (index % 6 == 2)
                copy_3x3_kernel(kernel_sobel_ny, kernel_touse);
            if (index % 6 == 3)
                copy_3x3_kernel(kernel_sobel_px, kernel_touse);
            if (index % 6 == 4)
                copy_3x3_kernel(kernel_sobel_nx, kernel_touse);
            if (index % 6 == 5)
                copy_3x3_kernel(kernel_laplacian, kernel_touse);

            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    kernel[in_channel * out_channels * 3 * 3 + out_channel * 3 * 3 + y * 3 + x] = kernel_touse[y * 3 + x];
                }
            }
        }
    }
}