#include <iostream>

#include <stdio.h>

#include "floatX.h"

int main(void)
{
    for(int i = 0; i < 32; i++)
    {
        for(int j = 0; j < 32; j++)
        {
            float a = i;
            float b = j;
            float c = a + b;
            float d = a * b;

            floatX<23, 8> a_mixed(a);
            floatX<23, 8> b_mixed(b);
            floatX<23, 8> c_mixed = a_mixed + b_mixed;
            floatX<23, 8> d_mixed = a_mixed * b_mixed;

            std::cout << a << "+" << b << "=" << c << std::endl;
            std::cout << a_mixed.to_float() << "+" << b_mixed.to_float() << "=" << c_mixed.to_float() << std::endl;

            std::cout << a << "*" << b << "=" << d << std::endl;
            std::cout << a_mixed.to_float() << "*" << b_mixed.to_float() << "=" << d_mixed.to_float() << std::endl;
        }
    }

    return 0;
}
