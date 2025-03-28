import torch
import torch.nn as nn
import torch.nn.functional as F


def param_to_bit(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def bit_to_param(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def fake_float_truncate(x: torch.Tensor,
                        e_bits_int: int,
                        m_bits_int: int,
                        scale_int: int) -> torch.Tensor:

    sign = x.sign()
    abs_x = x.abs().clamp(min=1e-45) / 2**scale_int

    # recover the floatint point representation
    # exponent \in {-2**7,..,2**7-1}
    # mantissa \in {1.0,...,2.0}

    exponent = torch.floor(torch.log2(abs_x)).clamp(min=1e-45)
    mantissa = abs_x / (2**exponent)

    # print("exp: ", exponent)
    # print("mantissa: ", mantissa)

    # truncate exponent
    # lets parameterize the exponent as a constant value + a variable value
    # the constant part is 2**7-1 in standar floating point, but we will learn it
    # the variable part \in {0,..,2**8-1}
    # lets say exponent = v_exponent - 2**(bits-1)-1 + c_exponent
    # so v_exponent = exponent + 2**(bits-1)-1 - c_exponent
    c_exponent = 0  # scale_int
    z_exponent = (2**(e_bits_int-1)-1)
    if e_bits_int == 0:
        z_exponent = 0
    v_exponent = exponent + z_exponent - c_exponent

    # print("v exponent: ", v_exponent)

    # the valriable part is clamped to the alloted bits
    q_min = torch.tensor(float(0)).to(x.device)
    q_max = 2**e_bits_int-1
    q_v_exponent = torch.clamp(v_exponent, q_min, q_max)
    q_exponent = q_v_exponent - z_exponent + c_exponent

    # print("q v exponent: ", q_v_exponent)
    # print("q exponent: ", q_exponent)

    # truncate mantissa
    # this just removes the less significant bits
    m_scale = 2.0 ** m_bits_int
    q_mantissa = torch.floor(mantissa * m_scale) / m_scale

    # print("q mantissa ", q_mantissa)

    # from quantized floatint point to float
    fq_x = sign * (2**q_exponent) * q_mantissa * 2**scale_int
    return fq_x


class FakeFloatFunction(torch.autograd.Function):
    """
    Custom autograd for 'fake-float' exponent+mantissa truncation.
    """
    @staticmethod
    def forward(ctx, x, e_bits_param, m_bits_param, scale_param):

        # save for backward
        ctx.save_for_backward(x, e_bits_param, m_bits_param, scale_param)

        # Round e_bits, m_bits to nearest integer for the forward pass
        e_bits_int = int(torch.round(param_to_bit(e_bits_param)).item())
        m_bits_int = int(torch.round(param_to_bit(m_bits_param)).item())
        s_int = int(torch.round(scale_param).item())

        out = fake_float_truncate(x, e_bits_int, m_bits_int, s_int)

        # if(m_bits_int == 0):
        #    print("input")
        #    print(x)
        #    print("output")
        #    print(out)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        x, e_bits_param, m_bits_param, scale_param = ctx.saved_tensors

        e_bits = param_to_bit(e_bits_param)
        m_bits = param_to_bit(m_bits_param)
        scale = scale_param
     
        e_bits_int = int(torch.round(e_bits).item())
        m_bits_int = int(torch.round(m_bits).item())
        scale_int = int(torch.round(scale).item())

        #print("shape x: ", x.shape)
        #print("shape grad_output: ", grad_output.shape)

        # 1) Gradient wrt x: straight-through
        grad_x = grad_outputs

        # 1) Gradient wrt x: approximate with central difference
        """
        grad_x = None
        if True:
            delta = 0.01            

            f_plus2  = fake_float_truncate(x + 2.0*delta, e_bits_int, m_bits_int, s_int)
            f_plus   = fake_float_truncate(x + 1.0*delta, e_bits_int, m_bits_int, s_int)
            f_minus  = fake_float_truncate(x - 1.0*delta, e_bits_int, m_bits_int, s_int)
            f_minus2 = fake_float_truncate(x - 2.0*delta, e_bits_int, m_bits_int, s_int)
        
            der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / (12.0 * delta)
            grad_x = grad_output * der
        """
                
        # 2) Gradient wrt e_bits: approximate with central difference
        grad_e_bits = None
        if e_bits_param.requires_grad:
            
            if(e_bits_int < 2):
                f_plus = fake_float_truncate(x,
                                             e_bits_int + 1,
                                             m_bits_int,
                                             scale_int)
                f_minus = fake_float_truncate(x,
                                              e_bits_int,
                                              m_bits_int,
                                              scale_int)
                der = (f_plus - f_minus)
            else:
                f_plus2 = fake_float_truncate(x,
                                              e_bits_int + 2,
                                              m_bits_int,
                                              scale_int)
                f_plus = fake_float_truncate(x,
                                             e_bits_int + 1,
                                             m_bits_int,
                                             scale_int)
                f_minus = fake_float_truncate(x,
                                              e_bits_int - 1,
                                              m_bits_int,
                                              scale_int)
                f_minus2 = fake_float_truncate(x,
                                               e_bits_int - 2,
                                               m_bits_int,
                                               scale_int)

                der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / 12.0

            grad_e_bits = grad_outputs * der * e_bits

        # 3) Gradient wrt m_bits: approximate with central difference
        grad_m_bits = None
        if m_bits_param.requires_grad:

            if (m_bits_int < 2):
                f_plus = fake_float_truncate(x,
                                             e_bits_int,
                                             m_bits_int + 1,
                                             scale_int)
                f_minus = fake_float_truncate(x,
                                              e_bits_int,
                                              m_bits_int,
                                              scale_int)
                der = (f_plus - f_minus)
            else:
                f_plus2 = fake_float_truncate(x,
                                              e_bits_int,
                                              m_bits_int + 2,
                                              scale_int)
                f_plus = fake_float_truncate(x,
                                             e_bits_int,
                                             m_bits_int + 1,
                                             scale_int)
                f_minus = fake_float_truncate(x,
                                              e_bits_int,
                                              m_bits_int - 1,
                                              scale_int)
                f_minus2 = fake_float_truncate(x,
                                               e_bits_int,
                                               m_bits_int - 2,
                                               scale_int)

                der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / 12.0

            grad_m_bits = grad_outputs * der * m_bits

        # 4) Gradient wrt m_bits: approximate with central difference
        grad_scale_bits = None
        if scale_param.requires_grad:

            f_plus2 = fake_float_truncate(x,
                                          e_bits_int,
                                          m_bits_int,
                                          scale_int + 2)
            f_plus = fake_float_truncate(x,
                                         e_bits_int,
                                         m_bits_int,
                                         scale_int + 1)
            f_minus = fake_float_truncate(x,
                                          e_bits_int,
                                          m_bits_int,
                                          scale_int - 1)
            f_minus2 = fake_float_truncate(x,
                                           e_bits_int,
                                           m_bits_int,
                                           scale_int - 2)

            der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / 12.0
            grad_scale_bits = grad_outputs * der

        return grad_x, grad_e_bits, grad_m_bits, grad_scale_bits


def test_fake_float_truncate():
    for i in range(100):
        in_test = (torch.rand(1) - 0.5)*100.0
        e_bits = int(torch.round(torch.rand(1)*10).item())
        m_bits = int(torch.round(torch.rand(1)*30).item())
        scale = int(torch.round((torch.rand(1) - 0.5)*5.0).item())
        out_test = fake_float_truncate(in_test, e_bits, m_bits, scale)
        print("e_bits ", e_bits, " m_bits ", m_bits, " scale ", scale)
        print("in: ", in_test, " out: ", out_test)


def fake_fixed_truncate2(x: torch.Tensor,
                         bits_int: int,
                         min: float,
                         max: float) -> torch.Tensor:

    qmin = 0
    qmax = 2**bits_int - 1

    scaled = (qmax - qmin) * (x - min) / (max - min) + qmin

    # from float to fixed point, and quantize accordingly
    q_x = torch.clamp(torch.round(scaled), qmin, qmax)

    # from quantized fixed point to float
    fq_x = (q_x - qmin) * (max - min) / (qmax - qmin) + min

    return fq_x


def fake_fixed_truncate(x: torch.Tensor,
                        bits_int: int,
                        scale_int: int,
                        zero_point_int: int) -> torch.Tensor:

    qmin = 0
    qmax = 2**bits_int - 1

    mantissa = x * 2**(scale_int + bits_int//2) + \
        zero_point_int + 2**(bits_int-1)

    # from float to fixed point, and quantize accordingly
    q_x = torch.clamp(torch.round(mantissa), qmin, qmax)

    # from quantized fixed point to float
    fq_x = (q_x - 2**(bits_int-1) - zero_point_int) / \
        2**(scale_int + bits_int//2)

    return fq_x


class FakeFixedFunction(torch.autograd.Function):
    """
    Custom autograd for 'fake-float' exponent+mantissa truncation.
    """
    @staticmethod
    def forward(ctx, x, bits_param, scale_param, zero_point_param):

        # save for backward
        ctx.save_for_backward(x, bits_param, scale_param, zero_point_param)

        # Round e_bits, m_bits to nearest integer for the forward pass
        bits_int = int(torch.round(param_to_bit(bits_param)).item())
        scale_int = int(torch.round(scale_param).item())
        zero_point_int = int(torch.round(zero_point_param).item())

        out = fake_fixed_truncate(x, bits_int, scale_int, zero_point_int)

        # print("input")
        # print(x)
        # print("output")
        # print(out)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        x, bits_param, scale_param, zero_point_param = ctx.saved_tensors

        bits = param_to_bit(bits_param)
        scale = scale_param
        zero_point = zero_point_param
    
        bits_int = int(torch.round(bits).item())
        scale_int = int(torch.round(scale).item())
        zero_point_int = int(torch.round(zero_point).item())

        #print("shape x: ", x.shape)
        #print("shape grad_output: ", grad_output.shape)

        # 1) Gradient wrt x: straight-through
        grad_x = grad_outputs

        # 1) Gradient wrt x: approximate with central difference
        """
        grad_x = None
        if True:
            delta = 0.01            

            f_plus2  = fake_float_truncate(x + 2.0*delta, e_bits_int, m_bits_int, s_int)
            f_plus   = fake_float_truncate(x + 1.0*delta, e_bits_int, m_bits_int, s_int)
            f_minus  = fake_float_truncate(x - 1.0*delta, e_bits_int, m_bits_int, s_int)
            f_minus2 = fake_float_truncate(x - 2.0*delta, e_bits_int, m_bits_int, s_int)

            der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / (12.0 * delta)
            grad_x = grad_output * der
        """
     
        # 2) Gradient wrt bits: approximate with central difference
        grad_bits = None
        if bits_param.requires_grad:
            if (bits_int < 2):
                f_plus = fake_fixed_truncate(x,
                                             bits_int + 1,
                                             scale_int,
                                             zero_point_int)
                f_minus = fake_fixed_truncate(x,
                                              bits_int,
                                              scale_int,
                                              zero_point_int)
                der = (f_plus - f_minus)
            else:
                f_plus2 = fake_fixed_truncate(x,
                                              bits_int + 2,
                                              scale_int,
                                              zero_point_int)
                f_plus = fake_fixed_truncate(x,
                                             bits_int + 1,
                                             scale_int,
                                             zero_point_int)
                f_minus = fake_fixed_truncate(x,
                                              bits_int - 1,
                                              scale_int,
                                              zero_point_int)
                f_minus2 = fake_fixed_truncate(x,
                                               bits_int - 2,
                                               scale_int,
                                               zero_point_int)

                der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / 12.0

            grad_bits = grad_outputs * der * bits

        # 3) Gradient wrt scale: approximate with central difference
        grad_scale_bits = None
        if scale_param.requires_grad:

            f_plus2 = fake_fixed_truncate(x,
                                          bits_int,
                                          scale_int + 2,
                                          zero_point_int)
            f_plus = fake_fixed_truncate(x,
                                         bits_int,
                                         scale_int + 1,
                                         zero_point_int)
            f_minus = fake_fixed_truncate(x,
                                          bits_int,
                                          scale_int - 1,
                                          zero_point_int)
            f_minus2 = fake_fixed_truncate(x,
                                           bits_int,
                                           scale_int - 2,
                                           zero_point_int)

            der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / 12.0

            grad_scale_bits = grad_outputs * der

        # 4) Gradient wrt m_bits: approximate with central difference
        grad_zero_point_bits = None
        if zero_point_param.requires_grad:

            f_plus2 = fake_fixed_truncate(x,
                                          bits_int,
                                          scale_int,
                                          zero_point_int + 2)
            f_plus = fake_fixed_truncate(x,
                                         bits_int,
                                         scale_int,
                                         zero_point_int + 1)
            f_minus = fake_fixed_truncate(x,
                                          bits_int,
                                          scale_int,
                                          zero_point_int - 1)
            f_minus2 = fake_fixed_truncate(x,
                                           bits_int,
                                           scale_int,
                                           zero_point_int - 2)

            der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / 12.0 
            grad_zero_point_bits = grad_outputs * der

        return grad_x, grad_bits, grad_scale_bits, grad_zero_point_bits


class FakeFixedFunction2(torch.autograd.Function):
    """
    Custom autograd for 'fake-float' exponent+mantissa truncation.
    """
    @staticmethod
    def forward(ctx, x, bits_param, min, max):

        # save for backward
        ctx.save_for_backward(x, bits_param, min, max)

        # Round e_bits, m_bits to nearest integer for the forward pass
        bits_int = int(torch.round(param_to_bit(bits_param)).item())
        out = fake_fixed_truncate2(x, bits_int, min, max)

        # print("input")
        # print(x)
        # print("output")
        # print(out)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        x, bits_param, min, max = ctx.saved_tensors

        bits = param_to_bit(bits_param)

        bits_int = int(torch.round(bits).item())

        # print("shape x: ", x.shape)
        # print("shape grad_output: ", grad_output.shape)

        # 1) Gradient wrt x: straight-through
        grad_x = grad_outputs

        # 1) Gradient wrt x: approximate with central difference
        """
        grad_x = None
        if True:
            delta = 0.01            

            f_plus2  = fake_float_truncate(x + 2.0*delta, e_bits_int, m_bits_int, s_int)
            f_plus   = fake_float_truncate(x + 1.0*delta, e_bits_int, m_bits_int, s_int)
            f_minus  = fake_float_truncate(x - 1.0*delta, e_bits_int, m_bits_int, s_int)
            f_minus2 = fake_float_truncate(x - 2.0*delta, e_bits_int, m_bits_int, s_int)

            der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / (12.0 * delta)
            grad_x = grad_output * der
        """

        # 2) Gradient wrt bits: approximate with central difference
        grad_bits = None
        if bits_param.requires_grad:
            if (bits_int < 2):
                f_plus = fake_fixed_truncate2(x,
                                              bits_int + 1,
                                              min,
                                              max)
                f_minus = fake_fixed_truncate2(x,
                                               bits_int,
                                               min,
                                               max)
                der = (f_plus - f_minus)
            else:
                f_plus2 = fake_fixed_truncate2(x,
                                               bits_int + 2,
                                               min,
                                               max)
                f_plus = fake_fixed_truncate2(x,
                                              bits_int + 1,
                                              min,
                                              max)
                f_minus = fake_fixed_truncate2(x,
                                               bits_int - 1,
                                               min,
                                               max)
                f_minus2 = fake_fixed_truncate2(x,
                                                bits_int - 2,
                                                min,
                                                max)

                der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / 12.0

            grad_bits = grad_outputs * der * bits

        return grad_x, grad_bits, None, None


def test_fake_fixed_truncate():
    for i in range(100):
        bits = 2  #int(torch.round(torch.rand(1)*32).item())
        scale = 0  #int(torch.round((torch.rand(1) - 0.5)*5.0).item())
        zero_point = 0  # int(torch.round((torch.rand(1) - 0.5)*0.0).item())
        in_test = (torch.rand(1)-0.5)*10.0
        # out_test = fake_fixed_truncate(in_test, bits, scale, zero_point)
        out_test = fake_fixed_truncate2(in_test, bits, -5.0, 5.0)
        print("bits ", bits, " scale ", scale, " zero_point ", zero_point)
        print("in: ", in_test, " out ", out_test)


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass: use the usual rounding
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        # Backward pass: pass the gradient unchanged (STE)
        return grad_outputs


class RoundFDE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass: use the usual rounding
        ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        # Backward pass: pass the gradient unchanged (STE)
        (input, ) = ctx.saved_tensors
        delta = 1.0
        f_plus2 = torch.round(input + 2*delta)
        f_plus = torch.round(input + 1*delta)
        f_minus = torch.round(input - 1*delta)
        f_minus2 = torch.round(input - 2*delta)
        # der = (f_plus - f_minus)/2.0
        der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / (12.0 * delta)

        return der * grad_outputs


class RoundSIG(torch.autograd.Function):
    """
    Custom autograd function that does a hard round in forward,
    but uses a sigmoid-based approximation for the backward pass.
    """

    @staticmethod
    def forward(ctx, input, alpha=10.0):
        """
        Forward pass: returns torch.round(input).
        """
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Backward pass: approximate the gradient of round(x)
        with the derivative of a sigmoid centered at the fractional midpoint (0.5).
        """
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha

        # Fractional part
        frac = input - torch.floor(input)

        # Sigmoid of (fractional_part - 0.5), scaled by alpha
        s = torch.sigmoid(alpha * (frac - 0.5))

        # Derivative of sigmoid = alpha * s * (1 - s)
        grad_input = alpha * s * (1 - s) * grad_outputs
        return grad_input, None  # alpha is not a tensor that requires grad


def diff_round(x):
    return RoundSTE.apply(x)
    # return RoundFDE.apply(x)
    # return RoundSIG.apply(x)


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass uses standard floor
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        # Straight-through pass: just return the gradient as-is
        return grad_outputs


class FloorFDE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass: use the usual rounding
        ctx.save_for_backward(input)
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        # Backward pass: pass the gradient unchanged (STE)
        (input, ) = ctx.saved_tensors
        delta = 1.0
        f_plus2 = torch.floor(input + 2*delta)
        f_plus = torch.floor(input + 1*delta)
        f_minus = torch.floor(input - 1*delta)
        f_minus2 = torch.floor(input - 2*delta)
        # der = (f_plus - f_minus)/2.0
        der = (-f_plus2 + 8*f_plus - 8*f_minus + f_minus2) / (12.0 * delta)

        return der * grad_outputs


def diff_floor(input):
    return FloorSTE.apply(input)
    # return FloorFDE.apply(input)


class MinMaxObserver(nn.Module):
    def __init__(self):
        super().__init__()
        # We store running min/max
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        # You could also store averaging stats, etc.

    def forward(self, x):
        # Update running min/max
        # self.min_val = torch.min(self.min_val, x.detach().min())
        # self.max_val = torch.max(self.max_val, x.detach().max())
        self.min_val = x.detach().min()
        self.max_val = x.detach().max()
        return x  # Just pass through


class FixedPointFakeQuantize(nn.Module):
    def __init__(self, observer, bits=32, requires_grad=False):
        super().__init__()
        self.observer = observer
        self.bits = nn.Parameter(torch.tensor(float(bits)), requires_grad=requires_grad)

    def forward(self, x):

        b_int = torch.clamp(diff_round(self.bits), 1, 32)

        # 1) Get min/max from observer
        min_val = self.observer.min_val
        max_val = self.observer.max_val

        # If they're not valid, skip
        # if min_val >= max_val:
        #    return x

        # 2) Compute scale and zero_point
        # For an unsigned 4-bit range, we can hold values 0..15
        # qmin, qmax = 0, (1 << b_int) - 1  # e.g. 0..15
        qmin, qmax = torch.tensor(float(0)), 2**b_int - 1  # e.g. 0..15

        qmin = qmin.to(x.device)
        # qmax = qmax.to(x.device)
        max_val = max_val.to(x.device)
        min_val = min_val.to(x.device)

        # Typical formula for scale/zero-point:
        scale = (max_val - min_val) / float(qmax - qmin)
        zero_point = qmin - diff_round(min_val / scale)

        # 3) Quantize (in floating point)
        # clamp to range of [qmin, qmax]
        q_x = torch.clamp(diff_round(x / scale + zero_point), qmin, qmax)

        # 4) Dequantize back to float
        fq_x = (q_x - zero_point) * scale
        return fq_x

    def getBits(self):
        return [self.bits]

    def printParams(self):
        print("bits: ", self.bits.detach().item())


class FixedPointFakeQuantize2(nn.Module):
    def __init__(self, bits=32, requires_grad=False):
        super().__init__()
        self.bits = nn.Parameter(torch.tensor(float(bits)), requires_grad=requires_grad)
        self.scale = nn.Parameter(torch.tensor(float(bits//2)), requires_grad=requires_grad)
        self.zero_point = nn.Parameter(torch.tensor(float(2**(bits//2-1)-1)), requires_grad=requires_grad)

    def forward(self, x):

        bits_int = torch.clamp(diff_round(self.bits), 1, 32)
        scale_int = diff_round(self.scale)
        zero_point_int = diff_round(self.zero_point)

        qmin = torch.tensor(float(0)).to(x.device)
        qmax = 2**bits_int - 1  # e.g. 0..15

        #from float to fixed point, and quantize accordingly
        q_x = torch.clamp(diff_round(x * 2**scale_int + zero_point_int), qmin, qmax)

        # from quantized fixed point to float
        fq_x = (q_x - zero_point_int) / 2**scale_int

        return fq_x

    def getBits(self):
        return [self.bits]

    def printParams(self):
        print("bits: ", self.bits.detach().item())
        print("scale: ", self.scale.detach().item())
        print("zero point: ", self.zero_point.detach().item())


class FloatingPointFakeQuantize(nn.Module):
    def __init__(self, m_bits=23, e_bits=8, requires_grad=False):
        super().__init__()
        self.e_bits = nn.Parameter(torch.tensor(float(e_bits)), requires_grad=requires_grad)
        self.m_bits = nn.Parameter(torch.tensor(float(m_bits)), requires_grad=requires_grad)
        self.scale = nn.Parameter(torch.tensor(float(0)), requires_grad=requires_grad)

    def forward(self, x):

        e_bits_int = torch.clamp(diff_round(self.e_bits), 0, 32)
        m_bits_int = torch.clamp(diff_round(self.m_bits), 1, 32)
        scale_int = diff_round(self.scale)

        sign = x.sign()
        abs_x = x.abs().clamp(min=1e-45)

        # recover the floatint point representation
        # exponent \in {-2**7,..,2**7-1}
        # mantissa \in {1.0,...,2.0}

        exponent = diff_floor(torch.log2(abs_x)).clamp(min=1e-45)
        mantissa = abs_x / (2**exponent)

        # truncate exponent
        # lets parameterize the exponent as a constant value + a variable value
        # the constant part is 2**7-1 in standar floating point, but we will learn it
        # the variable part \in {0,..,2**8-1}
        # lets say exponent = v_exponent - 2**(bits-1)-1 + c_exponent
        # so v_exponent = exponent + 2**(bits-1)-1 - c_exponent
        c_exponent = scale_int
        v_exponent = exponent + (2**(e_bits_int-1)-1) - c_exponent

        # the valriable part is clamped to the alloted bits
        q_min = torch.tensor(float(0)).to(x.device)
        q_max = 2**e_bits_int-1
        q_exponent = torch.clamp(v_exponent, q_min, q_max) - (2**(e_bits_int-1)-1) + c_exponent

        # truncate mantissa
        # this just removes the less significant bits
        m_scale = 2.0 ** m_bits_int
        q_mantissa = diff_floor(mantissa * m_scale) / m_scale

        # from quantized floatint point to float
        fq_x = sign * (2**q_exponent) * q_mantissa
        return fq_x

    def getBits(self):
        return [self.e_bits, self.m_bits]

    def printParams(self):
        print("e_bits: ", self.e_bits.detach().item())
        print("m_bits: ", self.m_bits.detach().item())
        print("scale: ", self.scale.detach().item())


class QuantWrapper(nn.Module):
    def __init__(self, module, optimizeQuant=False):
        super().__init__()
        self.module = module
        self.observer = MinMaxObserver()
        self.fake_quant_input = FixedPointFakeQuantize(self.observer, requires_grad=optimizeQuant)
        self.fake_quant_weight = FixedPointFakeQuantize(self.observer, requires_grad=optimizeQuant)
        # self.fake_quant_input = FixedPointFakeQuantize2(requires_grad=optimizeQuant)
        # self.fake_quant_weight = FixedPointFakeQuantize2(requires_grad=optimizeQuant)
        # self.fake_quant_input = FloatingPointFakeQuantize(requires_grad=optimizeQuant)
        # self.fake_quant_weight = FloatingPointFakeQuantize(requires_grad=optimizeQuant)

    def forward(self, x):
        x = self.observer(x)
        x = self.fake_quant_input(x)
        w = self.fake_quant_weight(self.module.weight)
        b = self.module.bias
        if isinstance(self.module, nn.Conv2d):
            return F.conv2d(x, w, b, stride=self.module.stride, padding=self.module.padding, dilation=self.module.dilation, groups=self.module.groups)
        elif isinstance(self.module, nn.Linear):
            return F.linear(x, w, b)
        else:
            return self.module(x)

    def getBits(self):
        return self.fake_quant_input.getBits() + self.fake_quant_weight.getBits()
        # return self.fake_quant_weight.getBits()

    def printQuantParams(self):
        print("input quant params: ")
        self.fake_quant_input.printParams()
        print("weight quant params: ")
        self.fake_quant_weight.printParams()


class QuantWrapperFloatingPoint(nn.Module):
    def __init__(self, module, e_bits=5, m_bits=10, optimizeQuant=False):
        super().__init__()
        self.module = module

        self.input_e_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(e_bits))), requires_grad=optimizeQuant)
        self.input_m_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(m_bits))), requires_grad=optimizeQuant)
        self.input_scale = nn.Parameter(torch.tensor(0.0), requires_grad=optimizeQuant)

        self.weight_e_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(e_bits))), requires_grad=optimizeQuant)
        self.weight_m_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(m_bits))), requires_grad=optimizeQuant)
        self.weight_scale = nn.Parameter(torch.tensor(0.0), requires_grad=optimizeQuant)

        # self.bias_e_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(e_bits))), requires_grad=False)
        # self.bias_m_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(m_bits))), requires_grad=False)
        # self.bias_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        # self.output_e_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(e_bits))), requires_grad=optimizeQuant)
        # self.output_m_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(m_bits))), requires_grad=optimizeQuant)
        # self.output_scale = nn.Parameter(torch.tensor(0.0), requires_grad=optimizeQuant)

    def forward(self, x):

        x = FakeFloatFunction.apply(x,   self.input_e_bits_param, self.input_m_bits_param, self.input_scale)

        if hasattr(self.module, 'weight') and self.module.weight != None:
            w = FakeFloatFunction.apply(self.module.weight, self.weight_e_bits_param, self.weight_m_bits_param, self.weight_scale)
        else:
            w = None

        if hasattr(self.module, 'bias') and self.module.bias != None:
            b = self.module.bias  # FakeFloatFunction.apply(self.module.bias, self.bias_e_bits_param, self.bias_m_bits_param, self.bias_scale)
        else:
            b = None

        if isinstance(self.module, nn.Conv2d):
            out = F.conv2d(x, w, b, stride=self.module.stride, padding=self.module.padding, dilation=self.module.dilation, groups=self.module.groups)
        elif isinstance(self.module, nn.Linear):
            out = F.linear(x, w, b)
        # else:
        #    out = self.module(x)

        # out = FakeFloatFunction.apply(out, self.output_e_bits_param, self.output_m_bits_param, self.output_scale)

        return out

    def getBits(self):
        return [param_to_bit(self.input_e_bits_param) + param_to_bit(self.input_m_bits_param) + 1, param_to_bit(self.weight_e_bits_param) + param_to_bit(self.weight_m_bits_param) + 1]
        # return [param_to_bit(self.weight_e_bits_param) + param_to_bit(self.weight_m_bits_param) + 1, param_to_bit(self.bias_e_bits_param) + param_to_bit(self.bias_m_bits_param) + 1, param_to_bit(self.output_e_bits_param) + param_to_bit(self.output_m_bits_param) + 1]
        # return [param_to_bit(self.weight_e_bits_param) + param_to_bit(self.weight_m_bits_param) + 1, param_to_bit(self.output_e_bits_param) + param_to_bit(self.output_m_bits_param) + 1]

    def printQuantParams(self):
        print("input quant params: ")
        print("e bits: ", param_to_bit(self.input_e_bits_param).detach().item(), " m bits ", param_to_bit(self.input_m_bits_param).detach().item(), " scale ", self.input_scale.detach().item())
        print("weight quant params: ")
        print("e bits ", param_to_bit(self.weight_e_bits_param).detach().item(), " m bits ", param_to_bit(self.weight_m_bits_param).detach().item(), " scale ", self.weight_scale.detach().item())
        # print("bias quant params: ")
        # print("e bits ", param_to_bit(self.bias_e_bits_param).detach().item(), " m bits ", param_to_bit(self.bias_m_bits_param).detach().item(), " scale ", self.bias_scale.detach().item())
        # print("output quant params: ")
        # print("e bits: ", param_to_bit(self.output_e_bits_param).detach().item(), " m bits ", param_to_bit(self.output_m_bits_param).detach().item(), " scale ", self.output_scale.detach().item())


class QuantWrapperFixedPoint(nn.Module):
    def __init__(self, module, bits=32, optimizeQuant=False):
        super().__init__()
        self.module = module

        self.input_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(bits))),
                                             requires_grad=optimizeQuant)
        self.input_scale = nn.Parameter(torch.tensor(float(0)),
                                        requires_grad=optimizeQuant)
        self.input_zero_point = nn.Parameter(torch.tensor(float(0)),
                                             requires_grad=optimizeQuant)

        self.weight_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(bits))),
                                              requires_grad=optimizeQuant)
        self.weight_scale = nn.Parameter(torch.tensor(float(0)),
                                         requires_grad=optimizeQuant)
        self.weight_zero_point = nn.Parameter(torch.tensor(float(0)),
                                              requires_grad=optimizeQuant)

        self.bias_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(bits))),
                                            requires_grad=False)
        self.bias_scale = nn.Parameter(torch.tensor(float(0)),
                                       requires_grad=False)
        self.bias_zero_point = nn.Parameter(torch.tensor(float(0)),
                                            requires_grad=False)

        # self.output_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(bits))), requires_grad=optimizeQuant)
        # self.output_scale = nn.Parameter(torch.tensor(float(0)), requires_grad=optimizeQuant)
        # self.output_zero_point = nn.Parameter(torch.tensor(float(0)), requires_grad=optimizeQuant)

    def forward(self, x):

        x = FakeFixedFunction.apply(x, self.input_bits_param,
                                     self.input_scale, self.input_zero_point)

        if hasattr(self.module, 'weight') and self.module.weight != None:
            w = FakeFixedFunction.apply(self.module.weight,
                                         self.weight_bits_param,
                                         self.weight_scale,
                                         self.weight_zero_point)
        else:
            w = None

        if hasattr(self.module, 'bias') and self.module.bias != None:
            b = FakeFixedFunction.apply(self.module.bias, self.bias_bits_param, self.bias_scale, self.bias_zero_point)
        else:
            b = None

        if isinstance(self.module, nn.Conv2d):
            out = F.conv2d(x, w, b, stride=self.module.stride,
                           padding=self.module.padding,
                           dilation=self.module.dilation,
                           groups=self.module.groups)
        elif isinstance(self.module, nn.Linear):
            out = F.linear(x, w, b)
        # else:
        #    out = self.module(x)

        # out = FakeFixedFunction.apply(out, self.output_bits_param, self.output_scale, self.output_zero_point)    

        return out

    def getBits(self):
        return [param_to_bit(self.input_bits_param),
                param_to_bit(self.weight_bits_param)]
        # return [param_to_bit(self.weight_bits_param), param_to_bit(self.bias_bits_param), param_to_bit(self.output_bits_param)]
        # return [param_to_bit(self.weight_bits_param), param_to_bit(self.output_bits_param)]

    def printQuantParams(self):
        print("input quant params: ")
        print("bits: ", param_to_bit(self.input_bits_param).detach().item(), " scale ", self.input_scale.detach().item(), " zero point ", self.input_zero_point.detach().item())
        print("weight quant params: ")
        print("bits: ", param_to_bit(self.weight_bits_param).detach().item(), " scale ", self.weight_scale.detach().item(), " zero point ", self.weight_zero_point.detach().item())
        print("bias quant params: ")
        print("bits: ", param_to_bit(self.bias_bits_param).detach().item(), " scale ", self.bias_scale.detach().item(), " zero point ", self.bias_zero_point.detach().item())
        # print("output quant params: ")
        # print("bits: ", param_to_bit(self.output_bits_param).detach().item(), " scale ", self.output_scale.detach().item(), " zero point ", self.output_zero_point.detach().item())


class QuantWrapperFixedPoint2(nn.Module):
    def __init__(self, module, weight_bits=32, act_bits=32, optimizeQuant=False):
        super().__init__()
        self.module = module

        self.input_observer = MinMaxObserver()
        self.weight_observer = MinMaxObserver()
        self.bias_observer = MinMaxObserver()

        self.input_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(act_bits))),
                                             requires_grad=optimizeQuant)
        self.weight_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(weight_bits))),
                                              requires_grad=optimizeQuant)
        self.bias_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(weight_bits))),
                                            requires_grad=False)

        # self.output_bits_param = nn.Parameter(bit_to_param(torch.tensor(float(bits))), requires_grad=optimizeQuant)
        # self.output_scale = nn.Parameter(torch.tensor(float(0)), requires_grad=optimizeQuant)
        # self.output_zero_point = nn.Parameter(torch.tensor(float(0)), requires_grad=optimizeQuant)

    def forward(self, x):

        x = self.input_observer(x)
        x = FakeFixedFunction2.apply(x,
                                     self.input_bits_param,
                                     self.input_observer.min_val,
                                     self.input_observer.max_val)

        if hasattr(self.module, 'weight') and self.module.weight != None:
            w = self.weight_observer(self.module.weight)
            w = FakeFixedFunction2.apply(w,
                                         self.weight_bits_param,
                                         self.weight_observer.min_val,
                                         self.weight_observer.max_val)
        else:
            w = None

        if hasattr(self.module, 'bias') and self.module.bias != None:
            b = self.bias_observer(self.module.bias)
            b = FakeFixedFunction2.apply(b,
                                         self.bias_bits_param,
                                         self.bias_observer.min_val,
                                         self.bias_observer.max_val)
        else:
            b = None

        if isinstance(self.module, nn.Conv2d):
            out = F.conv2d(x, w, b, stride=self.module.stride,
                           padding=self.module.padding,
                           dilation=self.module.dilation,
                           groups=self.module.groups)
        elif isinstance(self.module, nn.Linear):
            out = F.linear(x, w, b)
        # else:
        #    out = self.module(x)

        # out = FakeFixedFunction.apply(out, self.output_bits_param, self.output_scale, self.output_zero_point)    

        return out

    def getBits(self):
        return [param_to_bit(self.input_bits_param),
                param_to_bit(self.weight_bits_param)]
        # return [param_to_bit(self.weight_bits_param), param_to_bit(self.bias_bits_param), param_to_bit(self.output_bits_param)]
        # return [param_to_bit(self.weight_bits_param), param_to_bit(self.output_bits_param)]

    def printQuantParams(self):
        print("input quant params: ")
        print("bits: ", param_to_bit(self.input_bits_param).detach().item(), " scale ", self.input_scale.detach().item(), " zero point ", self.input_zero_point.detach().item())
        print("weight quant params: ")
        print("bits: ", param_to_bit(self.weight_bits_param).detach().item(), " scale ", self.weight_scale.detach().item(), " zero point ", self.weight_zero_point.detach().item())
        print("bias quant params: ")
        print("bits: ", param_to_bit(self.bias_bits_param).detach().item(), " scale ", self.bias_scale.detach().item(), " zero point ", self.bias_zero_point.detach().item())
        # print("output quant params: ")
        # print("bits: ", param_to_bit(self.output_bits_param).detach().item(), " scale ", self.output_scale.detach().item(), " zero point ", self.output_zero_point.detach().item())
