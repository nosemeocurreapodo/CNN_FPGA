#include "conv2D_3x3_float64_28x28.h"

int conv2D_3x3_float64_28x28(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, float weights[3 * 3])
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    return conv2D_3x3<float, conv_packet, 28, 28, 1>(input, output, weights);
}