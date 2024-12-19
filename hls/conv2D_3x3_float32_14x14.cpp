#include "conv2D_3x3_float32_14x14.h"

int conv2D_3x3_float32_14x14(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, float weights[3 * 3])
{
#pragma HLS INTERFACE mode=axis register_mode=both port=input
#pragma HLS INTERFACE mode=axis register_mode=both port=output
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    return conv2D_3x3<float, conv_packet, 14, 14, 1>(input, output, weights);
}