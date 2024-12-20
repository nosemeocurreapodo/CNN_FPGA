#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"

template <typename data_type, typename packet_type, int in_height, int in_width>
int MaxPooling2D(hls::stream<packet_type> &input, hls::stream<packet_type> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    shift_mat2<data_type, width> shift_reg;

main_y_loop:
    for (int y = 0; y < height; y++)
    {
    main_x_loop:
        for (int x = 0; x < width; x++)
        {
            // #pragma HLS PIPELINE II=1

            packet_type in_packet;
            input.read(in_packet);
            data_type in_data = data_type(in_packet.data);
            // if (in_packet.last == 1)
            //     last_was_read = true;

            shift_reg.shift_down(in_data);

            mat2<data_type> data = shift_reg.getMat();
            data_type max_val = data.getMax();

            packet_type out_packet;
            out_packet.data = float(max_val);
            out_packet.keep = -1;
            out_packet.strb = -1;
            if (x == width - 1 && y == height - 1)
                out_packet.last = true;
            else
                out_packet.last = false;

            if (x > 0 && y > 0 && (x % 2 != 0) && (y % 2 != 0))
            {
                output.write(out_packet);
            }
        }
    }

    return 0;
}
