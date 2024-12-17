#include "conv2D_3x3_float32_28x28.h"

int conv2D_3x3_float32_28x28(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, float weights[3 * 3])
{
#pragma HLS INTERFACE mode=axis register_mode=both port=input
#pragma HLS INTERFACE mode=axis register_mode=both port=output
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    return conv2D_3x3<float, conv_packet, 28, 28>(input, output, weights);
}