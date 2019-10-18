def msra_init(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight)
            # Modified by lzh @ 201707251408:
            # <<< Old:
            # if m.bias:
            # >>> New:
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            # Modified by lzh @ 201707241734:
            # <<< Old:
            # if m.bias:
            # >>> New:
            if m.bias is not None:
            # --- End
                init.constant(m.bias, 0)