#include "Linear_mixed_512x512.h"

int Linear_mixed_512x512(hls::stream<weight_packet_type> &weights_s, hls::stream<bias_packet_type> &bias_s, hls::stream<input_packet_type> &input_s, hls::stream<output_packet_type> &output_s)
{
#pragma HLS INTERFACE axis port = weights_s
#pragma HLS INTERFACE axis port = bias_s
#pragma HLS INTERFACE axis port = input_s
#pragma HLS INTERFACE axis port = output_s
#pragma HLS INTERFACE s_axilite port = return

    return Linear_base<weight_data_type, weight_packet_type, bias_data_type, bias_packet_type, input_data_type, input_packet_type, output_data_type, output_packet_type, 512, 512>(weights_s, bias_s, input_s, output_s);
}