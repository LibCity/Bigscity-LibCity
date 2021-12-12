import torch.nn as nn


class TemporalCONV(nn.Module):
    def __init__(self, input_fileter, out_time_fileter, time_strides, kernel_size=(1, 3)):
        self.time_conv = nn.Conv2d(input_fileter, out_time_fileter, kernel_size=kernel_size,
                                   stride=(1, time_strides), padding=(0, 1))

    def forward(self, inputs):
        # (B, T, N , F_in) -> (B, F_in, N, T) 用(1,3)的卷积核去做->(B, F_out', N, T')
        # F_in=input_filter,F_out'=nb_time_filter
        time_conv_output = self.time_conv(inputs.permute(0, 3, 2, 1))

        time_conv_output = time_conv_output.permute(0, 3, 2, 1)

        return time_conv_output
