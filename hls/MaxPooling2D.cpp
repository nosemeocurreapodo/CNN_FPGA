#include "MaxPooling2D.h"

int MaxPooling2D(hls::stream<input_packet_type> &input, hls::stream<output_packet_type> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    return MaxPooling2D_base<input_data_type, input_packet_type, output_data_type, output_packet_type, 224, 224>(input, output);
}