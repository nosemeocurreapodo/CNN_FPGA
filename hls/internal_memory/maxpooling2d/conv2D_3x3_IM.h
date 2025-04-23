#include "hls_stream.h"
#include "conv2D_3x3_IM_params.h"

extern int TOP_NAME(hls::stream<packet_type> &input, hls::stream<packet_type> &output);