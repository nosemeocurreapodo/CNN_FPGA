#include "MaxPooling2D_float32_112x112.h"

int MaxPooling2D_float32_112x112(hls::stream<mp_packet> &input, hls::stream<mp_packet> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    return MaxPooling2D<float, mp_packet, 112, 112>(input, output);
}