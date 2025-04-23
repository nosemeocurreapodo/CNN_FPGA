#pragma once

#include "HLSLinearAlgebra/src/types.h"

template <typename type, int size>
class shift_register
{
public:
    shift_register()
    {
    }

    type shift_down(type val_in)
    {
        // #pragma HLS INLINE

        type val_out = data[0];

    shift_down_loop:
        for (int i = 0; i < size - 1; i++)
        {
            // #pragma HLS UNROLL

            data[i] = data[i + 1];
        }

        data[size - 1] = val_in;

        return val_out;
    }

private:
    type data[size];
};

template <typename type, int size>
class shift_mat3
{
public:
    shift_mat3()
    {
    }

    void shift_down(type val)
    {
        type m20 = matrix(2, 0);
        matrix(2, 0) = matrix(2, 1);
        matrix(2, 1) = matrix(2, 2);
        matrix(2, 2) = val;

        type line2_val = line2.shift_down(m20);

        type m10 = matrix(1, 0);
        matrix(1, 0) = matrix(1, 1);
        matrix(1, 1) = matrix(1, 2);
        matrix(1, 2) = line2_val;

        type line1_val = line1.shift_down(m10);

        // type m00 = matrix(0, 0);
        matrix(0, 0) = matrix(0, 1);
        matrix(0, 1) = matrix(0, 2);
        matrix(0, 2) = line1_val;
    }

    mat3<type> getMat()
    {
        return matrix;
    }

private:
    shift_register<type, size - 3> line1;
    shift_register<type, size - 3> line2;
    mat3<type> matrix;
};

template <typename type, int size>
class shift_mat2
{
public:
    shift_mat2()
    {
    }

    void shift_down(type val)
    {
        type m10 = matrix(1, 0);
        matrix(1, 0) = matrix(1, 1);
        matrix(1, 1) = val;

        type line_val = line.shift_down(m10);

        // type m10 = matrix(1, 0);
        matrix(0, 0) = matrix(0, 1);
        matrix(0, 1) = line_val;
    }

    mat2<type> getMat()
    {
        return matrix;
    }

private:
    shift_register<type, size - 2> line;
    mat2<type> matrix;
};