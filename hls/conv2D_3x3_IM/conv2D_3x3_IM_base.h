#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "../common/types.h"

template <typename data_type, typename packet_type, int in_channels, int out_channels, int height, int width, int padding>
int conv2D_3x3_IM_base(hls::stream<packet_type> &input, hls::stream<packet_type> &output, const mat3<data_type> kernel[in_channels][out_channels])
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    int out_width = width - 2 + padding * 2;
    int out_height = height - 2 + padding * 2;

    data_type out_buffer[out_channels][out_height][out_width];

    shift_mat3<data_type, width + padding * 2> shift_reg;
    // #pragma HLS ARRAY_PARTITION variable=shift_reg.data dim=1 factor=5 type=cyclic
    //  #pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2

in_channel_loop:
    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
    in_y_loop:
        for (int y = -padding; y < height + padding; y++)
        {
        in_x_loop:
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

            main_out_channel_loop:
                for (int out_channel = 0; out_channel < out_channels; out_channel++)
                {
                    data_type conv = data.template mul_v2<data_type, data_type>(kernel[in_channel][out_channel]);

                    /*
                    packet_type out_packet;
                    out_packet.data = conv;
                    out_packet.keep = -1;
                    out_packet.strb = -1;
                    if (x == width + padding - 1 && y == height + padding - 1 && in_channel == in_channels - 1 && out_channel == out_channels - 1)
                        out_packet.last = true;
                    else
                        out_packet.last = false;

                    if (x >= 2 - padding && y >= 2 - padding)
                        output.write(out_packet);
                    */

                    if (x >= 2 - padding && y >= 2 - padding)
                    {
                        if (out_channel)
                            out_buffer[out_channel][y - 2 - padding][x - 2 - padding] = conv;
                        else
                            out_buffer[out_channel][y - 2 - padding][x - 2 - padding] += conv;
                    }
                }
            }
        }
    }

out_channel_loop:
    for (int out_channel = 0; out_channel < out_channels; out_channel++)
    {
    out_y_loop:
        for (int y = 0; y < out_height; y++)
        {
        out_x_loop:
            for (int x = 0; x < out_width; x++)
            {
                data_type out_data = out_buffer[out_channel][y][x];

                packet_type out_packet;
                out_packet.data = out_data;
                out_packet.keep = -1;
                out_packet.strb = -1;
                if (x == out_width - 1 && y == out_height - 1 && out_channel == out_channels - 1)
                    out_packet.last = true;
                else
                    out_packet.last = false;

                output.write(out_packet);
            }
        }
    }

    return 0;
}
