#include "hls_stream.h"

#include "../data_types_dict.h"
#include "maxpooling2d_params.h"
#include "maxpooling2d_base.h"

int TOP_NAME(hls::stream<in_packet_type> &input, hls::stream<out_packet_type> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    return MaxPooling2dBase<in_data_type, out_data_type, in_packet_type, out_packet_type, in_channels, in_height, in_width>(input, output);
}
