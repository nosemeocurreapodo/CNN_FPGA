#include "conv2D_3x3_float32_56x56.h"

int conv2D_3x3_float32_56x56(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, float weights[3 * 3])
{
#pragma HLS INTERFACE mode=axis register_mode=both port=input
#pragma HLS INTERFACE mode=axis register_mode=both port=output
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    return conv2D_3x3<float, conv_packet, 56, 56, 1>(input, output, weights);
}