#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"

// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
// #include "floatX.h"

#ifndef W_DATA_TYPE
using w_data_type = ap_fixed<8, 4>;
#else
#if W_DATA_TYPE == 1
using w_data_type = ap_fixed<1, 1>;
#elif W_DATA_TYPE == 2
using w_data_type = ap_fixed<2, 1>;
#elif W_DATA_TYPE == 4
using w_data_type = ap_fixed<4, 2>;
#elif W_DATA_TYPE == 8
using w_data_type = ap_fixed<8, 4>;
#elif W_DATA_TYPE == 16
using w_data_type = ap_fixed<16, 8>;
#elif W_DATA_TYPE == 32
using w_data_type = ap_fixed<32, 16>;
#else
#error "Unsupported W_DATA_TYPE"
#endif
#endif

#ifndef B_DATA_TYPE
using b_data_type = ap_fixed<8, 4>;
#else
#if B_DATA_TYPE == 1
using b_data_type = ap_fixed<1, 1>;
#elif B_DATA_TYPE == 2
using b_data_type = ap_fixed<2, 1>;
#elif B_DATA_TYPE == 4
using b_data_type = ap_fixed<4, 2>;
#elif B_DATA_TYPE == 8
using b_data_type = ap_fixed<8, 4>;
#elif B_DATA_TYPE == 16
using b_data_type = ap_fixed<16, 8>;
#elif B_DATA_TYPE == 32
using b_data_type = ap_fixed<32, 16>;
#else
#error "Unsupported B_DATA_TYPE"
#endif
#endif

#ifndef IN_DATA_TYPE
using in_data_type = ap_fixed<8, 4>;
#else
#if IN_DATA_TYPE == 1
using in_data_type = ap_fixed<1, 1>;
#elif IN_DATA_TYPE == 2
using in_data_type = ap_fixed<2, 1>;
#elif IN_DATA_TYPE == 4
using in_data_type = ap_fixed<4, 2>;
#elif IN_DATA_TYPE == 8
using in_data_type = ap_fixed<8, 4>;
#elif IN_DATA_TYPE == 16
using in_data_type = ap_fixed<16, 8>;
#elif IN_DATA_TYPE == 32
using in_data_type = ap_fixed<32, 16>;
#else
#error "Unsupported IN_DATA_TYPE"
#endif
#endif

#ifndef OUT_DATA_TYPE
using out_data_type = ap_fixed<8, 4>;
#else
#if OUT_DATA_TYPE == 1
using out_data_type = ap_fixed<1, 1>;
#elif OUT_DATA_TYPE == 2
using out_data_type = ap_fixed<2, 1>;
#elif OUT_DATA_TYPE == 4
using out_data_type = ap_fixed<4, 2>;
#elif OUT_DATA_TYPE == 8
using out_data_type = ap_fixed<8, 4>;
#elif OUT_DATA_TYPE == 16
using out_data_type = ap_fixed<16, 8>;
#elif OUT_DATA_TYPE == 32
using out_data_type = ap_fixed<32, 16>;
#else
#error "Unsupported OUT_DATA_TYPE"
#endif
#endif

using in_packet_type = hls::axis<in_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false>;
using out_packet_type = hls::axis<out_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false>;