#pragma once

#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"

template <int size>
int count_leading_zeros(ap_int<size> mantissa)
{
    //count leading zeros
    int leading_zeros = 0;
    for(int i = size-1; i >= 0; i--)
    {
        if(mantissa[i] == 0)
            leading_zeros++;
        else
            break;
    }
    return leading_zeros;
}

template <int mantissa_size, int exponent_size>
struct floatX
{
    floatX()
    {
        
    }

    floatX(float c)
    {
        float fp = c;

        bool s = 0;
        if(fp < 0.0)
            s = 1;
        
        float abs_fp = abs(fp);

        if(abs_fp == 0.0)
        {
            sign = 0;
            mantissa = 0;
            exponent = 0;
        }
        else
        {
            int exp = int(hls::log2(abs_fp));
            int quo = hls::pow(2, exp);
            float man = abs_fp / quo;    

            sign = s;
            exponent = hls::pow(2, (exponent_size - 1)) - 1 + exp;
            mantissa = man*hls::pow(2, mantissa_size); 
        }
    }

    float to_float()
    {
        float result;
        if(exponent == 0 && mantissa == 0)
        {
            result = 0.0;
        }
        else 
        { 
            ap_int<mantissa_size+2> un_mantissa = mantissa;
            un_mantissa[mantissa_size] = 1; //leading bit
            un_mantissa[mantissa_size+1] = 0; //leave space for sign

            float un_exponent = float(exponent) - hls::pow(2, exponent_size - 1) - 1;

            result = float(un_mantissa)*hls::pow(float(2.0), un_exponent);
            if(sign < 0)
                result = -result;
        }
        return result;
    }

    floatX operator+(floatX c)
    {
        floatX result;

        ap_int<mantissa_size+2> mantissa_1 = mantissa;
        ap_int<exponent_size> exponent_1 = exponent;
        mantissa_1[mantissa_size] = 1; //leading bit
        mantissa_1[mantissa_size+1] = 0; //leave space for sign
        
        ap_int<mantissa_size+2> mantissa_2 = c.mantissa;
        ap_int<exponent_size> exponent_2 = c.exponent;
        mantissa_2[mantissa_size] = 1; //leading bit
        mantissa_2[mantissa_size+1] = 0; //leave space for sign

        if(exponent_1 < exponent_2)
        {
            ap_int<mantissa_size+2> aux_mantissa = mantissa_1;
            ap_int<exponent_size> aux_exponent = exponent_1;
            mantissa_1 = mantissa_2;
            exponent_1 = exponent_2;
            mantissa_2 = aux_mantissa;
            exponent_2 = aux_exponent;
        }

        ap_int<exponent_size> exponent_diff = exponent_1 - exponent_2;

        if(sign == 1)
            mantissa_1 = -mantissa_1;
        if(c.sign == 1)
            mantissa_2 = -mantissa_2;

        mantissa_2 = mantissa_2 << exponent_diff;

        ap_int<mantissa_size+3> res_mantissa = mantissa_1 + mantissa_2;

        bool sign = 0;
        if(res_mantissa < 0)
        {
            sign = 1;
            res_mantissa = -res_mantissa;
        }

        int leading_zeros = count_leading_zeros(res_mantissa);

        result.sign = sign;
        result.exponent = exponent_1 + leading_zeros;
        result.mantissa = res_mantissa(1, mantissa_size);

        return result;
    }

    floatX operator*(floatX c)
    {
        floatX result;
        
        ap_int<mantissa_size+1> unnorm_mantissa_1 = mantissa;
        unnorm_mantissa_1[mantissa_size] = 1; //leading bit
        
        ap_int<mantissa_size+1> unnorm_mantissa_2 = c.mantissa;
        unnorm_mantissa_2[mantissa_size] = 1; //leading bit

        ap_int<(mantissa_size+1)*2> res_mantissa = unnorm_mantissa_1*unnorm_mantissa_2;

        int leading_zeros = count_leading_zeros<(mantissa_size+1)*2>(res_mantissa);

        res_mantissa = res_mantissa << leading_zeros;

        result.sign = sign ^ c.sign;
        result.mantissa = res_mantissa(1, mantissa_size);
        result.exponent = exponent + c.exponent + leading_zeros;

        return result;
    }

    floatX operator-(floatX c)
    {
        floatX negated = c;
        //netaged.sign = not netaged.sign;
        return negated;
    }

    void operator=(floatX c)
    {
        sign = c.sign;
        exponent = c.exponent;
        mantissa = c.mantissa;
    }

    //ap_fixed<mantissa_size, mantissa_size, AP_RND> mantissa;
    //ap_fixed<exponent_size, exponent_size, AP_RND> exponent;

    bool sign;
    ap_int<mantissa_size> mantissa;
    ap_int<exponent_size> exponent;
};

