from math import sqrt
import torch

def codegen_tensor_product_left_right(n_s):
    graph   = torch.fx.Graph()
    tracer  = torch.fx.proxy.GraphAppendingTracer(graph)
    x1s     = torch.fx.Proxy(graph.placeholder('x1', torch.Tensor), tracer=tracer)
    weights = torch.fx.Proxy(graph.placeholder('w',  torch.Tensor), tracer=tracer)
    result  = torch.einsum("zuw,zu->zw", weights.reshape((-1, n_s, n_s)), x1s) / sqrt(n_s)
    graph.output(result.node, torch.Tensor)
    graph.lint()
    graphmod = torch.fx.GraphModule(torch.nn.Module(), graph, class_name="tp_forward")
    return graphmod


class TensorProduct(torch.nn.Module):
    def __init__(self, n_s):
        super().__init__()
        self.n_s = n_s
        self.weight_numel = n_s * n_s
        self._compiled_main_left_right = codegen_tensor_product_left_right(n_s)
        output_mask = torch.ones(n_s)
        self.register_buffer('output_mask', output_mask)
        self.register_buffer('weight', torch.Tensor())
        self._profiling_str = str(self)

    def forward(self, x, y, weight):
        assert x.shape[-1] == self.n_s, "Incorrect last dimension for x"
        assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
        return self._compiled_main_left_right(x, weight)
