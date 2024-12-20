#include "MaxPooling2D_float32_14x14.h"

int MaxPooling2D_float32_14x14(hls::stream<mp_packet> &input, hls::stream<mp_packet> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    return MaxPooling2D<float, mp_packet, 14, 14>(input, output);
}