#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"
#include "MaxPooling2D.h"

// typedef ap_axis<32, 2, 5, 6> packet;
// typedef hls::axis<float, 0, 0, 0> packet;
// typedef hls::axis_data<float, AXIS_ENABLE_KEEP|AXIS_ENABLE_LAST> packet;

typedef hls::axis<float, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> mp_packet;

extern int MaxPooling2D_float32_56x56(hls::stream<mp_packet> &input, hls::stream<mp_packet> &output);
