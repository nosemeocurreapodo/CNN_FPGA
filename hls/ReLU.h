#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

#include "ReLU_base.h"

typedef ap_fixed<4, 1> data_type;
typedef hls::axis<data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> packet_type;

extern int ReLU(hls::stream<packet_type> &input1, hls::stream<packet_type> &input2, int &data_size);
