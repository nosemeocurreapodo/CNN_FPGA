#include "convolution.h"


int convolution(hls::stream<packet> &input, hls::stream<packet> &output, int &in_width, float kernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE], int &kernel_size) 
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = in_width
#pragma HLS INTERFACE s_axilite port = kernel
#pragma HLS INTERFACE s_axilite port = kernel_size
#pragma HLS INTERFACE s_axilite port = return

    data_type in_buffer[MAX_WIDTH*MAX_KERNEL_SIZE];

    init_buffer_y_loop:
    for(int j = 0; j < kernel_size; j++)
    {
        init_buffer_x_loop:
        for (int i = 0; i < in_width; i++)
        {
            in_buffer[i + j*in_width] = data_type(0.0);
        }
    }

    data_type i_kernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable = i_kernel complete dim = 0

    init_kernel_y_loop:
    for(int j = 0; j < kernel_size; j++)
    {
        init_kernel_x_loop:
        for (int i = 0; i < kernel_size; i++)
        {
            i_kernel[i + j*kernel_size] = kernel[i + j*kernel_size];
        }
    }

    bool last_was_read = false;
    int in_data_counter = 0;
    int padding_data_counter = 0;

    main_loop:
    while(true)
    {
        #pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        data_input_loop:
        for(int j = 0; j < kernel_size; j++)
        {
            if(j != 0)
            {
                in_buffer[in_width - 1 + (j-1)*in_width] = in_buffer[j*in_width];
            }
            for (int i = 0; i < in_width; i++)
            {
                in_buffer[i + j*in_width] = in_buffer[i + 1 + j*in_width];
            }
        }
        
        data_type in_data = data_type(0.0);
        if(!last_was_read)
        {
            packet in_packet; 
            input.read(in_packet);
            in_data = data_type(in_packet.data);
            if(in_packet.last == 1)
                last_was_read = true;
            in_data_counter++;
        }
        else 
        {
            padding_data_counter++;
        }

        if(in_data_counter < in_width*(kernel_size-1)/2)
            continue;

        if(padding_data_counter > in_width*(kernel_size-1)/2)
            break;

		in_buffer[in_width*kernel_size - 1] = in_data;

        data_type conv_partial_1[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];

        mult_y_loop:
        for(int y = 0; y < kernel_size; y++)
        {
            mult_x_loop:
            for(int x = 0; x < kernel_size; x++)
            {
                conv_partial_1[x + y*kernel_size] = in_buffer[x + y*in_width]*i_kernel[x + y*kernel_size];
            }
        }

        add_1_y_loop:
        for(int i = 0; i < kernel_size*kernel_size; i+=2)

        data_type conv = data_type(0.0);

        conv_add_y_loop:
        for(int y = 0; y < kernel_size; y++)
        {
            for(int x = 0; x < kernel_size; x++)
            {
                conv += conv_partial[x + y*kernel_size];
            }
        }

        packet out_packet;
        out_packet.data = conv;
        output.write(out_packet);
    }

    return 1;
}