#include "MaxPooling2D.h"


int MaxPooling2D(hls::stream<mp2D_packet> &input, hls::stream<mp2D_packet> &output, int &in_width, int &in_height)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = in_width
#pragma HLS INTERFACE s_axilite port = in_height
#pragma HLS INTERFACE s_axilite port = return

    shift_register<mp2D_data_type, MP2D_MAX_WIDTH + 2> shift_reg;
//#pragma HLS ARRAY_PARTITION variable=shift_reg.data dim=1 factor=5 type=cyclic
    // #pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2

    mat3<mp2D_data_type> kernel;
#pragma HLS ARRAY_PARTITION variable = kernel.data dim = 0 type = complete

    // initialize buffers

    bool last_was_read = false;

init_buffer_loop:
    for (int i = 0; i < MP2D_MAX_WIDTH + 2 - in_width - 1; i++)
    {
        mp2D_data_type in_data = mp2D_data_type(0.0f);
        if (!last_was_read)
        {
            mp2D_packet in_packet;
            input.read(in_packet);
            in_data = mp2D_data_type(in_packet.data);
            if (in_packet.last == 1)
                last_was_read = true;
        }

        shift_reg.shift_down(in_data);
    }

    int out_counter = 0;

main_loop:
    for(int i = 0; i < in_width*in_height; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        mat2<mp2D_data_type> data = shift_reg.getMat2(in_width);
        mp2D_data_type max_val = data.getMax();

        mp2D_packet out_packet;
        out_packet.data = float(max_val);
        out_packet.keep = -1;
        out_packet.strb = -1;
        if(out_counter == in_width*in_height/4 - 1)
            out_packet.last = true;
        else
            out_packet.last = false;

        int y = int(i / in_width);
        int x = i - y*in_width;
        if(x % 2 == 0 && y % 2 == 0)
        {
            output.write(out_packet);
            out_counter++;
        }

        mp2D_data_type in_data = mp2D_data_type(0.0f);
        if (!last_was_read)
        {
            mp2D_packet in_packet;
            input.read(in_packet);
            in_data = mp2D_data_type(in_packet.data);
            if (in_packet.last == 1)
                last_was_read = true;
        }

        shift_reg.shift_down(in_data);
    }

    return 0;
}