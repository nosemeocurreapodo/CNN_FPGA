#include "conv2D_3x3_float32_32x28x28.h"

int conv2D_3x3_float32_32x28x28(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, float weights[32 * 3 * 3])
{
#pragma HLS INTERFACE mode = axis register_mode = both port = input
#pragma HLS INTERFACE mode = axis register_mode = both port = output
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    conv2D_Nx3x3<float, conv_packet, 32, 28, 28, 1>(input, output, weights);

    return 0;
}