#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"

template <typename data_type, typename packet_type, int height, int width, int padding>
int conv2D_3x3(hls::stream<packet_type> &input, hls::stream<packet_type> &output, float weights[3 * 3])
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    shift_mat3<data_type, width + padding*2> shift_reg;
    // #pragma HLS ARRAY_PARTITION variable=shift_reg.data dim=1 factor=5 type=cyclic
    //  #pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2

    mat3<data_type> kernel;
#pragma HLS ARRAY_PARTITION variable = kernel.data dim = 0 type = complete

    // initialize buffers

init_kernel_y_loop:
    for (int y = 0; y < 3; y++)
    {
#pragma HLS UNROLL
    init_kernel_x_loop:
        for (int x = 0; x < 3; x++)
        {
#pragma HLS UNROLL
            kernel.data[y][x] = data_type(weights[x + y * 3]);
        }
    }

main_y_loop:
    for (int y = -padding; y < height + padding; y++)
    {
    main_x_loop:
        for (int x = -padding; x < width + padding; x++)
        {
            // #pragma HLS PIPELINE II=1

            data_type in_data = data_type(0.0f);
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                packet_type in_packet;
                input.read(in_packet);
                in_data = data_type(in_packet.data);
            }
            // if (in_packet.last == 1)
            //     last_was_read = true;

            shift_reg.shift_down(in_data);

            mat3<data_type> data = shift_reg.getMat();
            data_type conv = data.mul_v2(kernel);

            packet_type out_packet;
            out_packet.data = float(conv);
            out_packet.keep = -1;
            out_packet.strb = -1;
            if (x == width + padding - 1 && y == height + padding - 1)
                out_packet.last = true;
            else
                out_packet.last = false;

            if (x >= 2 - padding && y >= 2 - padding)
                output.write(out_packet);
        }
    }

    return 0;
}
