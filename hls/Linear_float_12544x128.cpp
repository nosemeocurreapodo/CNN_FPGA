#include "Linear_12544x128.h"

int Linear_12544x128(hls::stream<linear_packet> &weights_s, hls::stream<linear_packet> &bias_s, hls::stream<linear_packet> &input_s, hls::stream<linear_packet> &output_s)
{
#pragma HLS INTERFACE axis port = weights_s
#pragma HLS INTERFACE axis port = bias_s
#pragma HLS INTERFACE axis port = input_s
#pragma HLS INTERFACE axis port = output_s
#pragma HLS INTERFACE s_axilite port = return

    return Linear<float, linear_packet, 12544, 128>(weights_s, bias_s, input_s, output_s);
}