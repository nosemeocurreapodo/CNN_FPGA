#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"

// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
// #include "floatX.h"
#include "Posit.h"

#ifndef W_DATA_TYPE
using w_data_type = ap_fixed<8, 4>;
#else
#if W_DATA_TYPE == 1
using w_data_type = ap_fixed<1, 1>;
#elif W_DATA_TYPE == 2
using w_data_type = ap_fixed<2, 1>;
#elif W_DATA_TYPE == 3
using w_data_type = ap_fixed<4, 2>;
#elif W_DATA_TYPE == 4
using w_data_type = ap_fixed<8, 4>;
#elif W_DATA_TYPE == 5
using w_data_type = ap_fixed<16, 8>;
#elif W_DATA_TYPE == 6
using w_data_type = ap_fixed<32, 16>;
#elif W_DATA_TYPE == 7
using w_data_type = float; // Use float for -1
#elif W_DATA_TYPE == 8
using w_data_type = double; // Use double for -2
#elif W_DATA_TYPE == 9
using w_data_type = ap_float<8, 4>; // Use ap_float for -3
#elif W_DATA_TYPE == 10
using w_data_type = ap_float<16, 8>; // Use ap_float for -4
#elif W_DATA_TYPE == 11
using w_data_type = ap_float<32, 16>; // Use ap_float for -5
#elif W_DATA_TYPE == 12
using w_data_type = Posit<8, 1>; // Use Posit for -6
#elif W_DATA_TYPE == 13
using w_data_type = Posit<16, 2>; // Use Posit for -7
#elif W_DATA_TYPE == 14
using w_data_type = Posit<32, 3>; // Use Posit for -8
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
#elif B_DATA_TYPE == 3
using b_data_type = ap_fixed<4, 2>;
#elif B_DATA_TYPE == 4
using b_data_type = ap_fixed<8, 4>;
#elif B_DATA_TYPE == 5
using b_data_type = ap_fixed<16, 8>;
#elif B_DATA_TYPE == 6
using b_data_type = ap_fixed<32, 16>;
#elif B_DATA_TYPE == 7
using b_data_type = float; // Use float for -1
#elif B_DATA_TYPE == 8
using b_data_type = double; // Use double for -2
#elif B_DATA_TYPE == 9
using b_data_type = ap_float<8, 4>; // Use ap_float for -3
#elif B_DATA_TYPE == 10
using b_data_type = ap_float<16, 8>; // Use ap_float for -4
#elif B_DATA_TYPE == 11
using b_data_type = ap_float<32, 16>; // Use ap_float for -5
#elif B_DATA_TYPE == 12
using b_data_type = Posit<8, 1>; // Use Posit for -6
#elif B_DATA_TYPE == 13
using b_data_type = Posit<16, 2>; // Use Posit for -7
#elif B_DATA_TYPE == 14
using b_data_type = Posit<32, 3>; // Use Posit for -8
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
#elif IN_DATA_TYPE == 3
using in_data_type = ap_fixed<4, 2>;
#elif IN_DATA_TYPE == 4
using in_data_type = ap_fixed<8, 4>;
#elif IN_DATA_TYPE == 5
using in_data_type = ap_fixed<16, 8>;
#elif IN_DATA_TYPE == 6
using in_data_type = ap_fixed<32, 16>;
#elif IN_DATA_TYPE == 7
using in_data_type = float; // Use float for -1
#elif IN_DATA_TYPE == 8
using in_data_type = double; // Use double for -2
#elif IN_DATA_TYPE == 9
using in_data_type = ap_float<8, 4>; // Use ap_float for -3
#elif IN_DATA_TYPE == 10
using in_data_type = ap_float<16, 8>; // Use ap_float for -4
#elif IN_DATA_TYPE == 11
using in_data_type = ap_float<32, 16>; // Use ap_float for -5
#elif IN_DATA_TYPE == 12
using in_data_type = Posit<8, 1>; // Use Posit for -6
#elif IN_DATA_TYPE == 13
using in_data_type = Posit<16, 2>; // Use Posit for -7
#elif IN_DATA_TYPE == 14
using in_data_type = Posit<32, 3>; // Use Posit for -8
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
#elif OUT_DATA_TYPE == 3
using out_data_type = ap_fixed<4, 2>;
#elif OUT_DATA_TYPE == 4
using out_data_type = ap_fixed<8, 4>;
#elif OUT_DATA_TYPE == 5
using out_data_type = ap_fixed<16, 8>;
#elif OUT_DATA_TYPE == 6
using out_data_type = ap_fixed<32, 16>;
#elif OUT_DATA_TYPE == 7
using out_data_type = float; // Use float for -1
#elif OUT_DATA_TYPE == 8
using out_data_type = double; // Use double for -2
#elif OUT_DATA_TYPE == 9
using out_data_type = ap_float<8, 4>; // Use ap_float for -3
#elif OUT_DATA_TYPE == 10
using out_data_type = ap_float<16, 8>; // Use ap_float for -4
#elif OUT_DATA_TYPE == 11
using out_data_type = ap_float<32, 16>; // Use ap_float for -5
#elif OUT_DATA_TYPE == 12
using out_data_type = Posit<8, 1>; // Use Posit for -6
#elif OUT_DATA_TYPE == 13
using out_data_type = Posit<16, 2>; // Use Posit for -7
#elif OUT_DATA_TYPE == 14
using out_data_type = Posit<32, 3>; // Use Posit for -8
#else
#error "Unsupported OUT_DATA_TYPE"
#endif
#endif

using in_packet_type = hls::axis<in_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false>;
using out_packet_type = hls::axis<out_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false>;