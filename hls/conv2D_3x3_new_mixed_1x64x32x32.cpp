#include "conv2D_3x3_new_mixed_1x64x32x32.h"

int conv2D_3x3_new_mixed_1x64x32x32(hls::stream<input_packet_type> &input, hls::stream<output_packet_type> &output, hls::stream<weight_packet_type> &weights)
{
#pragma HLS INTERFACE mode=axis register_mode=both port=input
#pragma HLS INTERFACE mode=axis register_mode=both port=output
#pragma HLS INTERFACE mode=axis register_mode=both port=weights
#pragma HLS INTERFACE s_axilite port = return

    return conv2D_3x3_new_base<input_data_type, input_packet_type, output_data_type, output_packet_type, weight_data_type, weight_packet_type, 1, 64, 32, 32, 1>(input, output, weights);
}