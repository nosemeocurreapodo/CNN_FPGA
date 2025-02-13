#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"

template <typename input_type, typename input_packet, typename output_type, typename output_packet, typename weight_type, typename weights_packet, int in_channels, int out_channels, int in_height, int in_width, int padding>
int conv2D_3x3_new(hls::stream<input_packet> &input, hls::stream<output_type> &output, hls::stream<weight_packet> &weights)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE axis port = weights
#pragma HLS INTERFACE s_axilite port = return

    const int out_height = in_height + padding * 2 - 2;
    const int out_width = in_width + padding * 2 - 2;

    output_type output_data[out_height][out_width][out_channels]; // = {data_type(0.0)};
                                                                // #pragma HLS ARRAY_PARTITION variable=output_data dim=3 type=complete

    shift_mat3<input_type, in_width + padding * 2> shift_reg;
    // #pragma HLS ARRAY_PARTITION variable=shift_reg.data dim=1 factor=5 type=cyclic
    //  #pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2

    mat3<weight_type> kernel[in_channels][out_channels];
#pragma HLS ARRAY_PARTITION variable = kernel dim = 0 type = complete

init_out_channel_loop:
    for (int out_channel = 0; out_channel < out_channels; out_channel++)
    {
    init_in_channel_loop:
        for (int in_channel = 0; in_channel < in_channels; in_channel++)
        {
        init_kernel_y_loop:
            for (int y = 0; y < 3; y++)
            {
                // #pragma HLS UNROLL
            init_kernel_x_loop:
                for (int x = 0; x < 3; x++)
                {
                    // #pragma HLS UNROLL
                    weight_packet in_packet;
                    weights.read(in_packet);
                    weight_type in_data = data_type(in_packet.data);

                    kernel[in_channel][out_channel].data[y][x] = in_data;
                }
            }
        }
    }

main_in_channel_loop:
    for (int in_channel = 0; in_channel < in_channels; in_channel++)
    {
    main_y_loop:
        for (int y = -padding; y < in_height + padding; y++)
        {
        main_x_loop:
            for (int x = -padding; x < in_width + padding; x++)
            {
                // #pragma HLS PIPELINE II=1

                input_type in_data = data_type(0.0f);
                if (x >= 0 && x < in_width && y >= 0 && y < in_height)
                {
                    input_packet in_packet;
                    input.read(in_packet);
                    in_data = data_type(in_packet.data);
                }
                // if (in_packet.last == 1)
                //     last_was_read = true;

                shift_reg.shift_down(in_data);

            main_out_channel_loop:
                for (int out_channel = 0; out_channel < out_channels; out_channel++)
                {
                    mat3<input_type> data = shift_reg.getMat();
                    output_type conv = data.mul_v2(kernel[in_channel][out_channel]);

                    if (x >= 2 - padding && y >= 2 - padding)
                    {
                        if (in_channel == 0)
                            output_data[y - 2 + padding][x - 2 + padding][out_channel] = conv;
                        else
                            output_data[y - 2 + padding][x - 2 + padding][out_channel] += conv;
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
                output_packet out_packet;
                out_packet.data = float(output_data[y][x][out_channel]);
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
