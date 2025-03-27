#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "../common/types.h"

template <typename input_data_type, typename input_packet_type, typename output_data_type, typename output_packet_type, int in_channels, int in_height, int in_width>
int MaxPooling2_base(hls::stream<input_packet_type> &input, hls::stream<output_packet_type> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    shift_mat2<input_data_type, in_width> shift_reg;

main_channels_loop:
    for (int channel = 0; channel < in_channels; channel++)
    {
    main_y_loop:
        for (int y = 0; y < in_height; y++)
        {
        main_x_loop:
            for (int x = 0; x < in_width; x++)
            {
                // #pragma HLS PIPELINE II=1

                input_packet_type in_packet;
                input.read(in_packet);
                input_data_type in_data = input_data_type(in_packet.data);
                // if (in_packet.last == 1)
                //     last_was_read = true;

                shift_reg.shift_down(in_data);

                mat2<input_data_type> data = shift_reg.getMat();
                input_data_type max_val = data.getMax();

                output_packet_type out_packet;
                out_packet.data = output_data_type(max_val);
                out_packet.keep = -1;
                out_packet.strb = -1;
                if (x == in_width - 1 && y == in_height - 1 && channel == in_channels - 1)
                    out_packet.last = true;
                else
                    out_packet.last = false;

                if (x > 0 && y > 0 && (x % 2 != 0) && (y % 2 != 0))
                {
                    output.write(out_packet);
                }
            }
        }
    }
    return 0;
}
