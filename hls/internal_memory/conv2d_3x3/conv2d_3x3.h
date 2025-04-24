#pragma once

#include "hls_stream.h"
#include "conv2d_3x3_params.h"

extern int test(hls::stream<in_packet_type> &input, hls::stream<out_packet_type> &output);