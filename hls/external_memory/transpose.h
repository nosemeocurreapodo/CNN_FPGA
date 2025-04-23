#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"

template <typename data_type, typename packet_type, int in_channels, int in_height, int in_width>
int transpose(hls::stream<packet_type> &input, hls::stream<packet_type> &output)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    data_type output_data[2][in_height][in_width] = {data_type(0.0)};

    int out_count = 0;

main_in_channel_loop:
    for (int in_channel = 0; in_channel < in_channels + 1; in_channel++)
    {
    main_y_loop:
        for (int y = 0; y < in_height; y++)
        {
        main_x_loop:
            for (int x = 0; x < in_width; x++)
            {
                // #pragma HLS PIPELINE II=1

                data_type in_data = data_type(0.0f);
                if (in_channel >= 0 && in_channel < in_channels)
                {
                    packet_type in_packet;
                    input.read(in_packet);
                    in_data = data_type(in_packet.data);
                }

                // if (in_packet.last == 1)
                //     last_was_read = true;

                output_data[in_channel % 2][y][x] = in_data;

                if (in_channel > 0)
                {
                    //input = channelxheightxwidth
                    //output = heightxwidthxchannel

                    int out_c = out_count % 2;
                    int out_y = 
                    int out_x = 

                    packet_type out_packet;
                    out_packet.data = float(output_data[out_c][out_y][out_x]);
                    out_packet.keep = -1;
                    out_packet.strb = -1;
                    if (x == out_width - 1 && y == out_height - 1 && out_channel == out_channels - 1)
                        out_packet.last = true;
                    else
                        out_packet.last = false;

                    output.write(out_packet);
                    out_count++;
                }
            }
        }
    }

    return 0;
}
