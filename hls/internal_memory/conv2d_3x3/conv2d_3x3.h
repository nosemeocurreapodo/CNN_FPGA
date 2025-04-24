#pragma once

#include "conv2d_3x3_params.h"

extern int Conv2d3x3(hls::stream<in_packet_type> &input, hls::stream<out_packet_type> &output);