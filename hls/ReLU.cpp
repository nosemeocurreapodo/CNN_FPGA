#include "ReLU.h"

int ReLU(hls::stream<packet_type> &input, hls::stream<packet_type> &output, int &data_size)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = data_size
#pragma HLS INTERFACE s_axilite port = return

    return ReLU_base<data_type, packet_type>(input, output, data_size);
}