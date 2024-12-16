#include "dot_float32.h"

int dot_float32(hls::stream<dot_packet> &input1, hls::stream<dot_packet> &input2, float &result, int &in_size)
{
#pragma HLS INTERFACE axis port = input1
#pragma HLS INTERFACE axis port = input2
#pragma HLS INTERFACE s_axilite port = result
#pragma HLS INTERFACE s_axilite port = in_size
#pragma HLS INTERFACE s_axilite port = return

    return dot<float, dot_packet, 16>(input1, input2, result, in_size);
}