#include "convolution.h"
#include "data_types.h"

template <typename type>
class kernel_buffer
{
    public:
	kernel_buffer()
	{
		reset();
	}

	void reset()
	{
	kernel_reset_y_loop:
		for (int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
            kernel_reset_x_loop:
            for(int x = 0; x < MAX_KERNEL_SIZE; x++)
		    {
//#pragma HLS unroll
			    data[y][x] = type(0.0);
		    }
        }
	}

	type data[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
};

template <typename type>
class conv_buffer
{
    public:
	conv_buffer()
	{

	}

	void shift_down(type val, int in_width, int kernel_size)
	{
#pragma HLS INLINE

        data_type first_values[MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = first_values complete dim = 0

        save_init_values_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll

            first_values[y] = data[y][0];
        }

        shift_reg_y_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll

            shift_reg_x_loop:
            for (int x = 0; x < MAX_WIDTH-1; x++)
            {
#pragma HLS unroll

                data[y][x] = data[y][x + 1];
            }
        }

        write_init_values_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE-1; y++)
        {
#pragma HLS unroll

            data[y][in_width-1] = first_values[y];
        }

        data[kernel_size-1][in_width-1] = val;
	}

	type dot_3(kernel_buffer<type> a, int in_width)
	{
#pragma HLS INLINE

        type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

        dot_v3_mul_y_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll
            dot_v3_mul_x_loop:
            for(int x =0; x < MAX_KERNEL_SIZE; x++)
            {
#pragma HLS unroll
                mul[y][x] = data[y][x]*a.data[y][x];
            }
        }

		type sum_lvl1_0  = mul[0][0] + mul[1][0];
		type sum_lvl1_1  = mul[0][1] + mul[1][1];
		type sum_lvl1_2  = mul[0][2] + mul[1][2];

		type sum_lvl2_0  = mul[2][0] + sum_lvl1_0;
		type sum_lvl2_1  = mul[2][1] + sum_lvl1_1;
		type sum_lvl2_2  = mul[2][2] + sum_lvl1_2;

		type sum_lvl3_0 = sum_lvl2_0  + sum_lvl2_1;

        type res = sum_lvl3_0 + sum_lvl2_2;

		return res;
	}

	type dot_5(kernel_buffer<type> a, int in_width)
	{
#pragma HLS INLINE

        type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

        dot_v3_mul_y_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll
            dot_v3_mul_x_loop:
            for(int x =0; x < MAX_KERNEL_SIZE; x++)
            {
#pragma HLS unroll
                mul[y][x] = data[y][x]*a.data[y][x];
            }
        }

		type sum_lvl1_0  = mul[0][0] + mul[1][0];
		type sum_lvl1_1  = mul[0][1] + mul[1][1];
		type sum_lvl1_2  = mul[0][2] + mul[1][2];
		type sum_lvl1_3  = mul[0][3] + mul[1][3];
		type sum_lvl1_4  = mul[0][4] + mul[1][4];

		type sum_lvl1_5 = mul[2][0] + mul[3][0];
		type sum_lvl1_6 = mul[2][1] + mul[3][1];
		type sum_lvl1_7 = mul[2][2] + mul[3][2];
		type sum_lvl1_8 = mul[2][3] + mul[3][3];
		type sum_lvl1_9 = mul[2][4] + mul[3][4];

		type sum_lvl2_0 = mul[4][0] + sum_lvl1_0;
		type sum_lvl2_1 = mul[4][1] + sum_lvl1_1;
		type sum_lvl2_2 = mul[4][2] + sum_lvl1_2;
		type sum_lvl2_3 = mul[4][3] + sum_lvl1_3;
		type sum_lvl2_4 = mul[4][4] + sum_lvl1_4;

		type sum_lvl3_0 = sum_lvl2_0 + sum_lvl1_5;
		type sum_lvl3_1 = sum_lvl2_1 + sum_lvl1_6;
		type sum_lvl3_2 = sum_lvl2_2 + sum_lvl1_7;
		type sum_lvl3_3 = sum_lvl2_3 + sum_lvl1_8;
		type sum_lvl3_4 = sum_lvl2_4 + sum_lvl1_9;

		type sum_lvl4_0 = sum_lvl3_2  + sum_lvl3_0;
		type sum_lvl4_1 = sum_lvl3_3  + sum_lvl3_1;

		type sum_lvl5_0 = sum_lvl4_0  + sum_lvl3_4;
		
        type res = sum_lvl5_0  + sum_lvl4_1;

		return res;
	}

	type dot_7(kernel_buffer<type> a, int in_width)
	{
#pragma HLS INLINE

        type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

        dot_v3_mul_y_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll
            dot_v3_mul_x_loop:
            for(int x =0; x < MAX_KERNEL_SIZE; x++)
            {
#pragma HLS unroll
                mul[y][x] = data[y][x]*a.data[y][x];
            }
        }

		type sum_lvl1_0  = mul[0][0] + mul[1][0];
		type sum_lvl1_1  = mul[0][1] + mul[1][1];
		type sum_lvl1_2  = mul[0][2] + mul[1][2];
		type sum_lvl1_3  = mul[0][3] + mul[1][3];
		type sum_lvl1_4  = mul[0][4] + mul[1][4];
		type sum_lvl1_5  = mul[0][5] + mul[1][5];
		type sum_lvl1_6  = mul[0][6] + mul[1][6];

		type sum_lvl1_7  = mul[2][0] + mul[3][0];
		type sum_lvl1_8  = mul[2][1] + mul[3][1];
		type sum_lvl1_9  = mul[2][2] + mul[3][2];
		type sum_lvl1_10 = mul[2][3] + mul[3][3];
		type sum_lvl1_11 = mul[2][4] + mul[3][4];
		type sum_lvl1_12 = mul[2][5] + mul[3][5];
		type sum_lvl1_13 = mul[2][6] + mul[3][6];

		type sum_lvl1_14 = mul[4][0] + mul[5][0];
		type sum_lvl1_15 = mul[4][1] + mul[5][1];
		type sum_lvl1_16 = mul[4][2] + mul[5][2];
		type sum_lvl1_17 = mul[4][3] + mul[5][3];
		type sum_lvl1_18 = mul[4][4] + mul[5][4];
		type sum_lvl1_19 = mul[4][5] + mul[5][5];
		type sum_lvl1_20 = mul[4][6] + mul[5][6];

		type sum_lvl2_0 = mul[6][0] + sum_lvl1_0;
		type sum_lvl2_1 = mul[6][1] + sum_lvl1_1;
		type sum_lvl2_2 = mul[6][2] + sum_lvl1_2;
		type sum_lvl2_3 = mul[6][3] + sum_lvl1_3;
		type sum_lvl2_4 = mul[6][4] + sum_lvl1_4;
		type sum_lvl2_5 = mul[6][5] + sum_lvl1_5;
		type sum_lvl2_6 = mul[6][6] + sum_lvl1_6;

		type sum_lvl2_7  = sum_lvl1_7  + sum_lvl1_14;
		type sum_lvl2_8  = sum_lvl1_8  + sum_lvl1_15;
		type sum_lvl2_9  = sum_lvl1_9  + sum_lvl1_16;
		type sum_lvl2_10 = sum_lvl1_10 + sum_lvl1_17;
		type sum_lvl2_11 = sum_lvl1_11 + sum_lvl1_18;
		type sum_lvl2_12 = sum_lvl1_12 + sum_lvl1_19;
		type sum_lvl2_13 = sum_lvl1_13 + sum_lvl1_20;

		type sum_lvl3_0 = sum_lvl2_7  + sum_lvl2_0;
		type sum_lvl3_1 = sum_lvl2_8  + sum_lvl2_1;
		type sum_lvl3_2 = sum_lvl2_9  + sum_lvl2_2;
		type sum_lvl3_3 = sum_lvl2_10 + sum_lvl2_3;
		type sum_lvl3_4 = sum_lvl2_11 + sum_lvl2_4;
		type sum_lvl3_5 = sum_lvl2_12 + sum_lvl2_5;
		type sum_lvl3_6 = sum_lvl2_13 + sum_lvl2_6;

		type sum_lvl4_0 = sum_lvl3_3  + sum_lvl3_0;
		type sum_lvl4_1 = sum_lvl3_4  + sum_lvl3_1;
		type sum_lvl4_2 = sum_lvl3_5  + sum_lvl3_2;

		type sum_lvl5_0 = sum_lvl4_2  + sum_lvl4_0;
		type sum_lvl5_1 = sum_lvl3_6  + sum_lvl4_1;

        type res = sum_lvl5_0 + sum_lvl5_1;

		return res;
	}

	type data[MAX_KERNEL_SIZE][MAX_WIDTH];
};

template <typename type>
class conv_buffer_v2
{
    public:
	conv_buffer_v2()
	{

	}

	void shift_down(type val, int in_width, int kernel_size)
	{
#pragma HLS INLINE

        shift_reg_y_loop:
        for(int i = 0; i < MAX_KERNEL_SIZE*MAX_WIDTH-1; i++)
        {
#pragma HLS unroll

            data[i] = data[i + 1];
        }

        data[MAX_KERNEL_SIZE*MAX_WIDTH-1] = val;
	}

	type dot_3(kernel_buffer<type> a, int in_width)
	{
#pragma HLS INLINE

        type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

        dot_3_mul_y_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll
            dot_3_mul_x_loop:
            for(int x =0; x < MAX_KERNEL_SIZE; x++)
            {
#pragma HLS unroll
                mul[y][x] = data[x + y*in_width]*a.data[y][x];
            }
        }

		type sum_lvl1_0  = mul[0][0] + mul[1][0];
		type sum_lvl1_1  = mul[0][1] + mul[1][1];
		type sum_lvl1_2  = mul[0][2] + mul[1][2];

		type sum_lvl2_0  = mul[2][0] + sum_lvl1_0;
		type sum_lvl2_1  = mul[2][1] + sum_lvl1_1;
		type sum_lvl2_2  = mul[2][2] + sum_lvl1_2;

		type sum_lvl3_0 = sum_lvl2_0  + sum_lvl2_1;

        type res = sum_lvl3_0 + sum_lvl2_2;

		return res;
	}

	type dot_5(kernel_buffer<type> a, int in_width)
	{
#pragma HLS INLINE

        type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

        dot_5_mul_y_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll
            dot_5_mul_x_loop:
            for(int x =0; x < MAX_KERNEL_SIZE; x++)
            {
#pragma HLS unroll
                mul[y][x] = data[x + y*in_width]*a.data[y][x];
            }
        }

		type sum_lvl1_0  = mul[0][0] + mul[1][0];
		type sum_lvl1_1  = mul[0][1] + mul[1][1];
		type sum_lvl1_2  = mul[0][2] + mul[1][2];
		type sum_lvl1_3  = mul[0][3] + mul[1][3];
		type sum_lvl1_4  = mul[0][4] + mul[1][4];

		type sum_lvl1_5 = mul[2][0] + mul[3][0];
		type sum_lvl1_6 = mul[2][1] + mul[3][1];
		type sum_lvl1_7 = mul[2][2] + mul[3][2];
		type sum_lvl1_8 = mul[2][3] + mul[3][3];
		type sum_lvl1_9 = mul[2][4] + mul[3][4];

		type sum_lvl2_0 = mul[4][0] + sum_lvl1_0;
		type sum_lvl2_1 = mul[4][1] + sum_lvl1_1;
		type sum_lvl2_2 = mul[4][2] + sum_lvl1_2;
		type sum_lvl2_3 = mul[4][3] + sum_lvl1_3;
		type sum_lvl2_4 = mul[4][4] + sum_lvl1_4;

		type sum_lvl3_0 = sum_lvl2_0 + sum_lvl1_5;
		type sum_lvl3_1 = sum_lvl2_1 + sum_lvl1_6;
		type sum_lvl3_2 = sum_lvl2_2 + sum_lvl1_7;
		type sum_lvl3_3 = sum_lvl2_3 + sum_lvl1_8;
		type sum_lvl3_4 = sum_lvl2_4 + sum_lvl1_9;

		type sum_lvl4_0 = sum_lvl3_2  + sum_lvl3_0;
		type sum_lvl4_1 = sum_lvl3_3  + sum_lvl3_1;

		type sum_lvl5_0 = sum_lvl4_0  + sum_lvl3_4;
		
        type res = sum_lvl5_0  + sum_lvl4_1;

		return res;
	}

	type dot_7(kernel_buffer<type> a, int in_width)
	{
#pragma HLS INLINE

        type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

        dot_7_mul_y_loop:
        for(int y = 0; y < MAX_KERNEL_SIZE; y++)
        {
#pragma HLS unroll
            dot_7_mul_x_loop:
            for(int x =0; x < MAX_KERNEL_SIZE; x++)
            {
#pragma HLS unroll
                mul[y][x] = data[x + y*in_width]*a.data[y][x];
            }
        }

		type sum_lvl1_0  = mul[0][0] + mul[1][0];
		type sum_lvl1_1  = mul[0][1] + mul[1][1];
		type sum_lvl1_2  = mul[0][2] + mul[1][2];
		type sum_lvl1_3  = mul[0][3] + mul[1][3];
		type sum_lvl1_4  = mul[0][4] + mul[1][4];
		type sum_lvl1_5  = mul[0][5] + mul[1][5];
		type sum_lvl1_6  = mul[0][6] + mul[1][6];

		type sum_lvl1_7  = mul[2][0] + mul[3][0];
		type sum_lvl1_8  = mul[2][1] + mul[3][1];
		type sum_lvl1_9  = mul[2][2] + mul[3][2];
		type sum_lvl1_10 = mul[2][3] + mul[3][3];
		type sum_lvl1_11 = mul[2][4] + mul[3][4];
		type sum_lvl1_12 = mul[2][5] + mul[3][5];
		type sum_lvl1_13 = mul[2][6] + mul[3][6];

		type sum_lvl1_14 = mul[4][0] + mul[5][0];
		type sum_lvl1_15 = mul[4][1] + mul[5][1];
		type sum_lvl1_16 = mul[4][2] + mul[5][2];
		type sum_lvl1_17 = mul[4][3] + mul[5][3];
		type sum_lvl1_18 = mul[4][4] + mul[5][4];
		type sum_lvl1_19 = mul[4][5] + mul[5][5];
		type sum_lvl1_20 = mul[4][6] + mul[5][6];

		type sum_lvl2_0 = mul[6][0] + sum_lvl1_0;
		type sum_lvl2_1 = mul[6][1] + sum_lvl1_1;
		type sum_lvl2_2 = mul[6][2] + sum_lvl1_2;
		type sum_lvl2_3 = mul[6][3] + sum_lvl1_3;
		type sum_lvl2_4 = mul[6][4] + sum_lvl1_4;
		type sum_lvl2_5 = mul[6][5] + sum_lvl1_5;
		type sum_lvl2_6 = mul[6][6] + sum_lvl1_6;

		type sum_lvl2_7  = sum_lvl1_7  + sum_lvl1_14;
		type sum_lvl2_8  = sum_lvl1_8  + sum_lvl1_15;
		type sum_lvl2_9  = sum_lvl1_9  + sum_lvl1_16;
		type sum_lvl2_10 = sum_lvl1_10 + sum_lvl1_17;
		type sum_lvl2_11 = sum_lvl1_11 + sum_lvl1_18;
		type sum_lvl2_12 = sum_lvl1_12 + sum_lvl1_19;
		type sum_lvl2_13 = sum_lvl1_13 + sum_lvl1_20;

		type sum_lvl3_0 = sum_lvl2_7  + sum_lvl2_0;
		type sum_lvl3_1 = sum_lvl2_8  + sum_lvl2_1;
		type sum_lvl3_2 = sum_lvl2_9  + sum_lvl2_2;
		type sum_lvl3_3 = sum_lvl2_10 + sum_lvl2_3;
		type sum_lvl3_4 = sum_lvl2_11 + sum_lvl2_4;
		type sum_lvl3_5 = sum_lvl2_12 + sum_lvl2_5;
		type sum_lvl3_6 = sum_lvl2_13 + sum_lvl2_6;

		type sum_lvl4_0 = sum_lvl3_3  + sum_lvl3_0;
		type sum_lvl4_1 = sum_lvl3_4  + sum_lvl3_1;
		type sum_lvl4_2 = sum_lvl3_5  + sum_lvl3_2;

		type sum_lvl5_0 = sum_lvl4_2  + sum_lvl4_0;
		type sum_lvl5_1 = sum_lvl3_6  + sum_lvl4_1;

        type res = sum_lvl5_0 + sum_lvl5_1;

		return res;
	}

	type data[MAX_KERNEL_SIZE*MAX_WIDTH];
};

template <typename type>
type dot_3(ap_shift_reg<type, MAX_WIDTH*MAX_KERNEL_SIZE> &data, kernel_buffer<type> &a, int in_width)
{
#pragma HLS INLINE

    type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

    dot_3_mul_y_loop:
    for(int y = 0; y < MAX_KERNEL_SIZE; y++)
    {
#pragma HLS unroll
        dot_3_mul_x_loop:
        for(int x =0; x < MAX_KERNEL_SIZE; x++)
        {
#pragma HLS unroll
            mul[y][x] = data.read(x + y*in_width)*a.data[y][x];
        }
    }

    type sum_lvl1_0  = mul[0][0] + mul[1][0];
    type sum_lvl1_1  = mul[0][1] + mul[1][1];
    type sum_lvl1_2  = mul[0][2] + mul[1][2];

    type sum_lvl2_0  = mul[2][0] + sum_lvl1_0;
    type sum_lvl2_1  = mul[2][1] + sum_lvl1_1;
    type sum_lvl2_2  = mul[2][2] + sum_lvl1_2;

    type sum_lvl3_0 = sum_lvl2_0  + sum_lvl2_1;

    type res = sum_lvl3_0 + sum_lvl2_2;

    return res;
}

template <typename type>
type dot_5(ap_shift_reg<type, MAX_WIDTH*MAX_KERNEL_SIZE> &data, kernel_buffer<type> a, int in_width)
{
#pragma HLS INLINE

    type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

    dot_5_mul_y_loop:
    for(int y = 0; y < MAX_KERNEL_SIZE; y++)
    {
#pragma HLS unroll
        dot_5_mul_x_loop:
        for(int x =0; x < MAX_KERNEL_SIZE; x++)
        {
#pragma HLS unroll
            mul[y][x] = data.read(x + y*in_width)*a.data[y][x];
        }
    }

    type sum_lvl1_0  = mul[0][0] + mul[1][0];
    type sum_lvl1_1  = mul[0][1] + mul[1][1];
    type sum_lvl1_2  = mul[0][2] + mul[1][2];
    type sum_lvl1_3  = mul[0][3] + mul[1][3];
    type sum_lvl1_4  = mul[0][4] + mul[1][4];

    type sum_lvl1_5 = mul[2][0] + mul[3][0];
    type sum_lvl1_6 = mul[2][1] + mul[3][1];
    type sum_lvl1_7 = mul[2][2] + mul[3][2];
    type sum_lvl1_8 = mul[2][3] + mul[3][3];
    type sum_lvl1_9 = mul[2][4] + mul[3][4];

    type sum_lvl2_0 = mul[4][0] + sum_lvl1_0;
    type sum_lvl2_1 = mul[4][1] + sum_lvl1_1;
    type sum_lvl2_2 = mul[4][2] + sum_lvl1_2;
    type sum_lvl2_3 = mul[4][3] + sum_lvl1_3;
    type sum_lvl2_4 = mul[4][4] + sum_lvl1_4;

    type sum_lvl3_0 = sum_lvl2_0 + sum_lvl1_5;
    type sum_lvl3_1 = sum_lvl2_1 + sum_lvl1_6;
    type sum_lvl3_2 = sum_lvl2_2 + sum_lvl1_7;
    type sum_lvl3_3 = sum_lvl2_3 + sum_lvl1_8;
    type sum_lvl3_4 = sum_lvl2_4 + sum_lvl1_9;

    type sum_lvl4_0 = sum_lvl3_2  + sum_lvl3_0;
    type sum_lvl4_1 = sum_lvl3_3  + sum_lvl3_1;

    type sum_lvl5_0 = sum_lvl4_0  + sum_lvl3_4;
    
    type res = sum_lvl5_0  + sum_lvl4_1;

    return res;
}

template <typename type>
type dot_7(ap_shift_reg<type, MAX_WIDTH*MAX_KERNEL_SIZE> &data, kernel_buffer<type> &a, int in_width)
{
#pragma HLS INLINE

    type mul[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable = mul complete dim = 0

    dot_v3_mul_y_loop:
    for(int y = 0; y < MAX_KERNEL_SIZE; y++)
    {
#pragma HLS unroll
        dot_v3_mul_x_loop:
        for(int x =0; x < MAX_KERNEL_SIZE; x++)
        {
#pragma HLS unroll
            mul[y][x] = data.read(MAX_WIDTH - x + (MAX_KERNEL_SIZE - y)*in_width)*a.data[MAX_KERNEL_SIZE - 1 - y][MAX_KERNEL_SIZE - 1 - x];
        }
    }

    type sum_lvl1_0  = mul[0][0] + mul[1][0];
    type sum_lvl1_1  = mul[0][1] + mul[1][1];
    type sum_lvl1_2  = mul[0][2] + mul[1][2];
    type sum_lvl1_3  = mul[0][3] + mul[1][3];
    type sum_lvl1_4  = mul[0][4] + mul[1][4];
    type sum_lvl1_5  = mul[0][5] + mul[1][5];
    type sum_lvl1_6  = mul[0][6] + mul[1][6];

    type sum_lvl1_7  = mul[2][0] + mul[3][0];
    type sum_lvl1_8  = mul[2][1] + mul[3][1];
    type sum_lvl1_9  = mul[2][2] + mul[3][2];
    type sum_lvl1_10 = mul[2][3] + mul[3][3];
    type sum_lvl1_11 = mul[2][4] + mul[3][4];
    type sum_lvl1_12 = mul[2][5] + mul[3][5];
    type sum_lvl1_13 = mul[2][6] + mul[3][6];

    type sum_lvl1_14 = mul[4][0] + mul[5][0];
    type sum_lvl1_15 = mul[4][1] + mul[5][1];
    type sum_lvl1_16 = mul[4][2] + mul[5][2];
    type sum_lvl1_17 = mul[4][3] + mul[5][3];
    type sum_lvl1_18 = mul[4][4] + mul[5][4];
    type sum_lvl1_19 = mul[4][5] + mul[5][5];
    type sum_lvl1_20 = mul[4][6] + mul[5][6];

    type sum_lvl2_0 = mul[6][0] + sum_lvl1_0;
    type sum_lvl2_1 = mul[6][1] + sum_lvl1_1;
    type sum_lvl2_2 = mul[6][2] + sum_lvl1_2;
    type sum_lvl2_3 = mul[6][3] + sum_lvl1_3;
    type sum_lvl2_4 = mul[6][4] + sum_lvl1_4;
    type sum_lvl2_5 = mul[6][5] + sum_lvl1_5;
    type sum_lvl2_6 = mul[6][6] + sum_lvl1_6;

    type sum_lvl2_7  = sum_lvl1_7  + sum_lvl1_14;
    type sum_lvl2_8  = sum_lvl1_8  + sum_lvl1_15;
    type sum_lvl2_9  = sum_lvl1_9  + sum_lvl1_16;
    type sum_lvl2_10 = sum_lvl1_10 + sum_lvl1_17;
    type sum_lvl2_11 = sum_lvl1_11 + sum_lvl1_18;
    type sum_lvl2_12 = sum_lvl1_12 + sum_lvl1_19;
    type sum_lvl2_13 = sum_lvl1_13 + sum_lvl1_20;

    type sum_lvl3_0 = sum_lvl2_7  + sum_lvl2_0;
    type sum_lvl3_1 = sum_lvl2_8  + sum_lvl2_1;
    type sum_lvl3_2 = sum_lvl2_9  + sum_lvl2_2;
    type sum_lvl3_3 = sum_lvl2_10 + sum_lvl2_3;
    type sum_lvl3_4 = sum_lvl2_11 + sum_lvl2_4;
    type sum_lvl3_5 = sum_lvl2_12 + sum_lvl2_5;
    type sum_lvl3_6 = sum_lvl2_13 + sum_lvl2_6;

    type sum_lvl4_0 = sum_lvl3_3  + sum_lvl3_0;
    type sum_lvl4_1 = sum_lvl3_4  + sum_lvl3_1;
    type sum_lvl4_2 = sum_lvl3_5  + sum_lvl3_2;

    type sum_lvl5_0 = sum_lvl4_2  + sum_lvl4_0;
    type sum_lvl5_1 = sum_lvl3_6  + sum_lvl4_1;

    type res = sum_lvl5_0 + sum_lvl5_1;

    return res;
}

int convolution(hls::stream<packet> &input, hls::stream<packet> &output, int &in_width, float kernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE], int &kernel_size) 
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = in_width
#pragma HLS INTERFACE s_axilite port = kernel
#pragma HLS INTERFACE s_axilite port = kernel_size
#pragma HLS INTERFACE s_axilite port = return

//conv_buffer_v2<data_type> in_buffer;
//#pragma HLS ARRAY_PARTITION variable=in_buffer.data dim=1 factor=10 type=block
//#pragma HLS ARRAY_PARTITION variable=in_buffer.data dim=1 type=cyclic factor=10
    static ap_shift_reg<data_type, MAX_WIDTH*MAX_KERNEL_SIZE> in_buffer;
//#pragma HLS ARRAY_PARTITION variable=in_buffer.data dim=1 type=complete
//#pragma HLS ARRAY_PARTITION variable=in_buffer.data dim=2 type=cyclic factor=2
//#pragma HLS ARRAY_PARTITION variable=in_buffer.data dim=2 type=complete

//#pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2
    kernel_buffer<data_type> i_kernel;
#pragma HLS ARRAY_PARTITION variable=i_kernel.data dim=0 type=complete

    //initialize buffers

    init_buffer_loop:
    for(int i = 0; i < MAX_KERNEL_SIZE*MAX_WIDTH; i++)
    {
        //in_buffer.shift_down(0, MAX_WIDTH, MAX_KERNEL_SIZE);
        in_buffer.shift(0, 1);
    }

    init_kernel_y_loop:
    for(int y = 0; y < MAX_KERNEL_SIZE; y++)
    {
        init_kernel_x_loop:
        for (int x = 0; x < MAX_KERNEL_SIZE; x++)
        {
            i_kernel.data[y][x] = kernel[x + y*MAX_KERNEL_SIZE];
        }
    }

    bool last_was_read = false;
    int data_counter = 0;
    int padding_data_counter = 0;

    main_loop:
    while(true)
    {
//#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        data_type in_data = data_type(0.0);
        if(!last_was_read)
        {
            packet in_packet; 
            input.read(in_packet);
            in_data = data_type(in_packet.data);
            if(in_packet.last == 1)
                last_was_read = true;
            data_counter++;
        }
        else 
        {
            padding_data_counter++;
        }

        if(data_counter < MAX_WIDTH*(kernel_size-1)/2)
            continue;

        if(padding_data_counter > MAX_WIDTH*(kernel_size-1)/2)
            break;

        //in_buffer.shift_down(in_data, in_width, kernel_size);
        in_buffer.shift(in_data, 1);

#if(MAX_KERNEL_SIZE == 3)
        //data_type conv = in_buffer.dot_3(i_kernel, in_width);
        data_type conv = dot_3(in_buffer, i_kernel, in_width);
#endif
#if(MAX_KERNEL_SIZE == 5)
        //data_type conv = in_buffer.dot_5(i_kernel, in_width);
        data_type conv = dot_5(in_buffer, i_kernel, in_width);
#endif
#if(MAX_KERNEL_SIZE == 7)
        //data_type conv = in_buffer.dot_7(i_kernel, in_width);
        data_type conv = dot_7(in_buffer, i_kernel, in_width);
#endif
        

        packet out_packet;
        out_packet.data = conv;
        output.write(out_packet);
    }

    return 1;
}