#include "hls_stream.h"
#include "conv2d_3x3_params.h"

extern int TOP_NAME(hls::stream<packet_type> &input, hls::stream<packet_type> &output);