#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
//#include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"

#define CONV_MAX_WIDTH 256

//typedef ap_fixed<24, 12, AP_RND> conv_data_type;
//typedef ap_float<32, 8> conv_data_type;
//typedef floatX<23, 8> conv_data_type;
typedef float conv_data_type;
//typedef half conv_data_type;
//typedef int conv_data_type;

//typedef ap_axis<32, 2, 5, 6> packet;
//typedef hls::axis<float, 0, 0, 0> packet;
//typedef hls::axis_data<float, AXIS_ENABLE_KEEP|AXIS_ENABLE_LAST> packet;

typedef  hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> conv_packet;

extern int conv2D_3x3(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, int &in_width, int &in_height, float kernel[9]);
