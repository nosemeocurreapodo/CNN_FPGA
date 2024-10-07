#pragma once

#include "ap_axi_sdata.h"
//#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "data_types.h"

#define MAX_KERNEL_SIZE 5
#define MAX_WIDTH 1024

#define CONV_BUFFER_SIZE 10

extern int convolution(hls::stream<packet> &input, hls::stream<packet> &output, int &in_width, float kernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE], int &kernel_size); 
