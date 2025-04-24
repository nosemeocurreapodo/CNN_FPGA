#include "hls_stream.h"

#include "maxpooling2d.h"
#include "maxpooling2dBase.h"

int MaxPooling2d(hls::stream<input_packet_type> &input, hls::stream<output_packet_type> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    return MaxPooling2dBase<input_data_type, output_data_type, input_packet_type, output_packet_type, in_channels, in_height, in_width>(input, output);
}