#pragma once

#include "maxpooling2d_params.h"

extern int MaxPooling2d(hls::stream<in_packet_type> &input, hls::stream<out_packet_type> &output);