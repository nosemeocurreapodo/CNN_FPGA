#include "Linear_16384x512.h"

int Linear_16384x512(hls::stream<weight_packet> &weights_s, hls::stream<bias_packet> &bias_s, hls::stream<input_packet> &input_s, hls::stream<output_packet> &output_s)
{
#pragma HLS INTERFACE axis port = weights_s
#pragma HLS INTERFACE axis port = bias_s
#pragma HLS INTERFACE axis port = input_s
#pragma HLS INTERFACE axis port = output_s
#pragma HLS INTERFACE s_axilite port = return

    return Linear<weight_type, weight_packet, bias_type, bias_packet, input_type, input_packet, output_type, output_packet, 16384, 512>(weights_s, bias_s, input_s, output_s);
}