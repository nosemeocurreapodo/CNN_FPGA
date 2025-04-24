#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "linalg.h"
#include "shift_registers.h"

template <typename w_data_type, typename in_data_type, typename out_data_type, typename in_packet_type, typename out_packet_type, bool use_relu, int batch_size, int in_channels, int out_channels, int in_height, int in_width, int padding>
int Conv2d3x3Base(hls::stream<in_packet_type> &input, hls::stream<out_packet_type> &output, const linalg::Mat3<w_data_type> kernel[in_channels][out_channels])
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    const int out_width = in_width - 2 + padding * 2;
    const int out_height = in_height - 2 + padding * 2;

    out_data_type out_buffer[out_channels][out_height][out_width];

    ShiftMat3<in_data_type, in_width + padding * 2> shift_reg;
    // #pragma HLS ARRAY_PARTITION variable=shift_reg.data dim=1 factor=5 type=cyclic
    //  #pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2

batch_size_loop:
    for (int batch = 0; batch < batch_size; batch++)
    {
    in_channel_loop:
        for (int in_channel = 0; in_channel < in_channels; in_channel++)
        {
        in_y_loop:
            for (int y = -padding; y < in_height + padding; y++)
            {
            in_x_loop:
                for (int x = -padding; x < in_width + padding; x++)
                {
                    // #pragma HLS PIPELINE II=1

                    in_data_type in_data = in_data_type(0.0f);
                    if (x >= 0 && x < in_width && y >= 0 && y < in_height)
                    {
                        in_packet_type in_packet;
                        input.read(in_packet);
                        in_data = in_packet.data;
                    }
                    // if (in_packet.last == 1)
                    //     last_was_read = true;

                    shift_reg.ShiftDown(in_data);
                    linalg::Mat3<in_data_type> data = shift_reg.GetMat();

                main_out_channel_loop:
                    for (int out_channel = 0; out_channel < out_channels; out_channel++)
                    {
                        // out_data_type conv = data.template mul_v2<out_data_type, w_data_type>(kernel[in_channel][out_channel]);
                        out_data_type conv = data.template conv<out_data_type, w_data_type>(kernel[in_channel][out_channel]);

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
                            if (in_channel == 0)
                                out_buffer[out_channel][y - 2 + padding][x - 2 + padding] = conv;
                            else
                                out_buffer[out_channel][y - 2 + padding][x - 2 + padding] += conv;
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
                    out_data_type out_data = out_buffer[out_channel][y][x];

                    if (use_relu)
                        if (out_data < out_data_type(0.0f))
                            out_data = out_data_type(0.0f);

                    out_packet_type out_packet;
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
    }

    return 0;
}
