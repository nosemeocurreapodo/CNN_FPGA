#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

#include "Linear.h"

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> linear_packet;

extern int Linear_12544x128(hls::stream<linear_packet> &weights_s, hls::stream<linear_packet> &bias_s, hls::stream<linear_packet> &input_s, hls::stream<linear_packet> &output_s);
