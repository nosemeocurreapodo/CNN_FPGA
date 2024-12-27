#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"
#include "conv2D_3x3_new.h"

// typedef ap_axis<32, 2, 5, 6> packet;
// typedef hls::axis<float, 0, 0, 0> packet;
// typedef hls::axis_data<float, AXIS_ENABLE_KEEP|AXIS_ENABLE_LAST> packet;

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> conv_packet;

extern int conv2D_3x3_new_1x32x28x28(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, float weights[9]);
