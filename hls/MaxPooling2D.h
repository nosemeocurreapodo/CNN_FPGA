#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"

template <typename data_type, typename packet_type, int width, int height>
int MaxPooling2D(hls::stream<packet_type> &input, hls::stream<packet_type> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    shift_mat2<data_type, width> shift_reg;
    // #pragma HLS ARRAY_PARTITION variable=shift_reg.data dim=1 factor=5 type=cyclic
    //  #pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2

    mat3<data_type> kernel;
#pragma HLS ARRAY_PARTITION variable = kernel.data dim = 0 type = complete

    // initialize buffers

    bool last_was_read = false;

init_buffer_loop:
    for (int i = 0; i < width + 2; i++)
    {
        data_type in_data = data_type(0.0f);
        if (!last_was_read)
        {
            packet_type in_packet;
            input.read(in_packet);
            in_data = data_type(in_packet.data);
            if (in_packet.last == 1)
                last_was_read = true;
        }

        shift_reg.shift_down(in_data);
    }

    int out_counter = 0;

main_loop:
    for (int i = 0; i < height * width; i++)
    {
        // #pragma HLS PIPELINE II=1

        mat2<data_type> data = shift_reg.getMat();
        data_type max_val = data.getMax();

        packet_type out_packet;
        out_packet.data = float(max_val);
        out_packet.keep = -1;
        out_packet.strb = -1;
        if (out_counter == width * height / 4 - 2)
            out_packet.last = true;
        else
            out_packet.last = false;

        int y = int(i / width);
        int x = i - y * width;
        if ((x % 2 == 0) && (y % 2 == 0))
        {
            output.write(out_packet);
            out_counter++;
        }

        data_type in_data = data_type(0.0f);
        if (!last_was_read)
        {
            packet_type in_packet;
            input.read(in_packet);
            in_data = data_type(in_packet.data);
            if (in_packet.last == 1)
                last_was_read = true;
        }

        shift_reg.shift_down(in_data);
    }

    return 0;
}
