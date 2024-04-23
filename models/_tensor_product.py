import warnings
from math import sqrt

import torch
from torch import fx

from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode



def codegen_tensor_product_left_right(n_s, path_weight) -> fx.GraphModule:
    graph = fx.Graph()

    # = Function definitions =
    tracer = fx.proxy.GraphAppendingTracer(graph)

    x1s = fx.Proxy(graph.placeholder('x1', torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder('x2', torch.Tensor), tracer=tracer)
    weights = fx.Proxy(graph.placeholder('w', torch.Tensor), tracer=tracer)

    empty = fx.Proxy(graph.call_function(torch.empty, ((),), dict(device='cpu')), tracer=tracer)
    output_shape = torch.broadcast_tensors(empty.expand(x1s.shape[:-1]),
                                           empty.expand(x2s.shape[:-1]),
                                           empty.expand(weights.shape[:-1]))[0].shape
    del empty

    # = Broadcast inputs =
    x1s, x2s, weights = x1s.broadcast_to(output_shape + (-1,)), \
                        x2s.broadcast_to(output_shape + (-1,)), \
                        weights.broadcast_to(output_shape + (-1,))

    output_shape = output_shape + (n_s,)

    x1s = x1s.reshape(-1, n_s)
    x2s = x2s.reshape(-1, 1)
    w = weights.reshape((-1, n_s, 1, n_s))

    result = torch.einsum("zuvw,zu,zv->zw", w, x1s, x2s)
    result = path_weight * result
    result = result.reshape(output_shape)

    graph.output(result.node, torch.Tensor)
    graph.lint()

    constants_root = torch.nn.Module()
    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_forward")
    return graphmod



@compile_mode('script')
class TensorProduct(CodeGenMixin, torch.nn.Module):
    def __init__(self, n_s):

        super().__init__()

        self._in1_dim = n_s
        self._in2_dim = 1

        graphmod_left_right = codegen_tensor_product_left_right(n_s, sqrt(1.0/n_s))
        graphmod_right = fx.Graph()
        graphmod_right.placeholder('x2', torch.Tensor)
        graphmod_right.placeholder('w', torch.Tensor)
        graphmod_right.call_function(
            torch._assert,
            args=(False, "`right` method is not compiled, set `compile_right` to True when creating the TensorProduct")
        )
        graphmod_right = fx.GraphModule(torch.nn.Module(), graphmod_right, class_name="tp_forward")
        self._codegen_register({
            "_compiled_main_left_right": graphmod_left_right,
            "_compiled_main_right": graphmod_right
        })

        self.weight_numel = n_s * n_s
        self.register_buffer('weight', torch.Tensor())

        output_mask = torch.ones(n_s)
        self.register_buffer('output_mask', output_mask)
        self._profiling_str = str(self)


    def forward(self, x, y, weight):
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"
        assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
        print(x.shape[:-1], y.shape[:-1])
        #print(y.shape)
        return self._compiled_main_left_right(x, y, weight)
