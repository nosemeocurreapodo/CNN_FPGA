#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

#define LINEAR_MAX_WIDTH 256

// typedef ap_fixed<24, 12, AP_RND> relu_data_type;
// typedef ap_float<32, 8> relu_data_type;
// typedef floatX<23, 8> relu_data_type;
typedef float linear_data_type;
// typedef half relu_data_type;
// typedef int relu_data_type;

// typedef ap_axis<32, 2, 5, 6> packet;
// typedef hls::axis<float, 0, 0, 0> packet;
// typedef hls::axis_data<float, AXIS_ENABLE_KEEP|AXIS_ENABLE_LAST> packet;

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> linear_packet;

extern int Linear(hls::stream<linear_packet> &input, hls::stream<linear_packet> &output, int &in_size, int &max_width, float weights[LINEAR_MAX_WIDTH]);
