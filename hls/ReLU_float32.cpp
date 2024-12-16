#include "ReLU_float32.h"

int ReLU_float32(hls::stream<relu_packet> &input, hls::stream<relu_packet> &output, int &data_size)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = data_size
#pragma HLS INTERFACE s_axilite port = return

    return ReLU<float, relu_packet>(input, output, data_size);
}