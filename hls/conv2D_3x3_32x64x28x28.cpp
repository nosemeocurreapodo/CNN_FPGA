#include "conv2D_3x3_32x64x28x28.h"

int conv2D_3x3_32x64x28x28(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, hls::stream<conv_packet> &weights)
{
#pragma HLS INTERFACE mode=axis register_mode=both port=input
#pragma HLS INTERFACE mode=axis register_mode=both port=output
#pragma HLS INTERFACE mode=axis register_mode=both port=weights
#pragma HLS INTERFACE s_axilite port = return

    return conv2D_3x3<float, conv_packet, 32, 64, 28, 28, 1>(input, output, weights);
}