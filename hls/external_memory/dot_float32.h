#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

#include "dot.h"

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> dot_packet;

extern int dot_float32(hls::stream<dot_packet> &input1, hls::stream<dot_packet> &input2, float &result, int &in_size);
