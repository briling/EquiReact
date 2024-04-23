import warnings
from math import sqrt
from typing import List, Optional, Union, Any, Callable
from collections import OrderedDict

import e3nn
import torch
#import torch.fx
from e3nn import o3
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from torch import fx
from e3nn.o3._tensor_product._instruction import Instruction


from opt_einsum_fx import optimize_einsums_full


def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)

def codegen_tensor_product_left_right(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> fx.GraphModule:
    graph = fx.Graph()

    # = Function definitions =
    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = OrderedDict()

    x1s = fx.Proxy(graph.placeholder('x1', torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder('x2', torch.Tensor), tracer=tracer)
    weights = fx.Proxy(graph.placeholder('w', torch.Tensor), tracer=tracer)

    empty = fx.Proxy(graph.call_function(torch.empty, ((),), dict(device='cpu')), tracer=tracer)
    if shared_weights:
        output_shape = torch.broadcast_tensors(empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1]))[0].shape
    else:
        output_shape = torch.broadcast_tensors(empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1]), empty.expand(weights.shape[:-1]))[0].shape
    del empty

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        outputs = x1s.new_zeros(output_shape + (irreps_out.dim,))

        graph.output(outputs.node, torch.Tensor)
        # Short circut
        return fx.GraphModule({}, graph, "tp_forward")

    # = Broadcast inputs =
    if shared_weights:
        x1s, x2s = x1s.broadcast_to(output_shape + (-1,)), x2s.broadcast_to(output_shape + (-1,))
    else:
        x1s, x2s, weights = x1s.broadcast_to(output_shape + (-1,)), x2s.broadcast_to(output_shape + (-1,)), weights.broadcast_to(output_shape + (-1,))

    output_shape = output_shape + (irreps_out.dim,)

    x1s = x1s.reshape(-1, irreps_in1.dim)
    x2s = x2s.reshape(-1, irreps_in2.dim)

    batch_numel = x1s.shape[0]

    # = Determine number of weights and reshape weights ==
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        weights = weights.reshape(-1, weight_numel)
    del weight_numel

    # = extract individual input irreps =
    # If only one input irrep, can avoid creating a view
    if len(irreps_in1) == 1:
        x1_list = [x1s.reshape(batch_numel, irreps_in1[0].mul, irreps_in1[0].ir.dim)]
    else:
        x1_list = [
            x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
        ]

    x2_list = []
    # If only one input irrep, can avoid creating a view
    if len(irreps_in2) == 1:
        x2_list.append(
            x2s.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].ir.dim)
        )
    else:
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list.append(
                x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            )

    # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
    z = '' if shared_weights else 'z'

    # Cache of input irrep pairs whose outer products (xx) have already been computed
    xx_dict = dict()

    # Current index in the flat weight tensor
    flat_weight_index = 0

    outputs = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        x1 = x1_list[ins.i_in1]
        x2 = x2_list[ins.i_in2]

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv', 'uvu<v', 'u<vw']

        if ins.has_weight:
            # Extract the weight from the flattened weight tensor
            w = weights[:, flat_weight_index:flat_weight_index + prod(ins.path_shape)].reshape((() if shared_weights else (-1,)) + tuple(ins.path_shape))
            flat_weight_index += prod(ins.path_shape)

        # Construct the general xx in case this instruction isn't specialized
        # If this isn't used, the dead code will get removed
        key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == 'uu':
                xx_dict[key] = torch.einsum('zui,zuj->zuij', x1, x2)
            else:
                xx_dict[key] = torch.einsum('zui,zvj->zuvij', x1, x2)
        xx = xx_dict[key]
        del key

        # Create a proxy & request for the relevant wigner w3j
        # If not used (because of specialized code), will get removed later.
        w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
        w3j = fx.Proxy(graph.get_attr(w3j_name), tracer=tracer)

        l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

        if ins.connection_mode == 'uvw':
            assert ins.has_weight
            if specialized_code and l1l2l3 == (0, 0, 0):
                result = torch.einsum(f"{z}uvw,zu,zv->zw", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
            elif specialized_code and mul_ir_in1.ir.l == 0:
                result = torch.einsum(f"{z}uvw,zu,zvj->zwj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_in2.ir.l == 0:
                result = torch.einsum(f"{z}uvw,zui,zv->zwi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_out.ir.l == 0:
                result = torch.einsum(f"{z}uvw,zui,zvi->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
            else:
                result = torch.einsum(f"{z}uvw,ijk,zuvij->zwk", w, w3j, xx)
        if ins.connection_mode == 'uvu':
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum(f"{z}uv,zu,zv->zu", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zu,zvj->zuj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zui,zv->zui", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zui,zvi->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}uv,ijk,zuvij->zuk", w, w3j, xx)
            else:
                # not so useful operation because v is summed
                result = torch.einsum("ijk,zuvij->zuk", w3j, xx)
        if ins.connection_mode == 'uvv':
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum(f"{z}uv,zu,zv->zv", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zu,zvj->zvj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zui,zv->zvi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zui,zvi->zv", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}uv,ijk,zuvij->zvk", w, w3j, xx)
            else:
                # not so useful operation because u is summed
                # only specialize out for this path
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum("zu,zv->zv", x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum("zu,zvj->zvj", x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum("zui,zv->zvi", x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum("zui,zvi->zv", x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum("ijk,zuvij->zvk", w3j, xx)
        if ins.connection_mode == 'uuw':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum(f"{z}uw,zu,zu->zw", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum(f"{z}uw,zu,zuj->zwj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum(f"{z}uw,zui,zu->zwi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum(f"{z}uw,zui,zui->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}uw,ijk,zuij->zwk", w, w3j, xx)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                result = torch.einsum("ijk,zuij->zk", w3j, xx)
        if ins.connection_mode == 'uuu':
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum(f"{z}u,zu,zu->zu", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and l1l2l3 == (1, 1, 1):
                    result = torch.einsum(
                        f"{z}u,zui->zui",
                        w,
                        torch.cross(x1, x2, dim=2)
                    ) / sqrt(2*3)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum(f"{z}u,zu,zuj->zuj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum(f"{z}u,zui,zu->zui", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum(f"{z}u,zui,zui->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}u,ijk,zuij->zuk", w, w3j, xx)
            else:
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum("zu,zu->zu", x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and l1l2l3 == (1, 1, 1):
                    result = torch.cross(x1, x2, dim=2) * (1.0 / sqrt(2*3))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum("zu,zuj->zuj", x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum("zui,zu->zui", x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum("zui,zui->zu", x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum("ijk,zuij->zuk", w3j, xx)
        if ins.connection_mode == 'uvuv':
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                result = torch.einsum(f"{z}uv,ijk,zuvij->zuvk", w, w3j, xx)
            else:
                # TODO implement specialized code
                result = torch.einsum("ijk,zuvij->zuvk", w3j, xx)
        if ins.connection_mode == 'uvu<v':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
            name = f"_triu_indices_{mul_ir_in1.mul}"
            constants[name] = torch.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
            i = fx.Proxy(graph.get_attr(name), tracer=tracer)
            xx = xx[:, i[0], i[1]]  # zuvij -> zwij
            if ins.has_weight:
                # TODO implement specialized code
                result = torch.einsum(f"{z}w,ijk,zwij->zwk", w, w3j, xx)
            else:
                # TODO implement specialized code
                result = torch.einsum("ijk,zwij->zwk", w3j, xx)
        if ins.connection_mode == 'u<vw':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            assert ins.has_weight
            name = f"_triu_indices_{mul_ir_in1.mul}"
            constants[name] = torch.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
            i = fx.Proxy(graph.get_attr(name), tracer=tracer)
            xx = xx[:, i[0], i[1]]  # zuvij -> zqij
            # TODO implement specialized code
            result = torch.einsum(f"{z}qw,ijk,zqij->zwk", w, w3j, xx)

        result = ins.path_weight * result

        outputs += [result.reshape(batch_numel, mul_ir_out.dim)]

        # Remove unused w3js:
        if len(w3j.node.users) == 0:
            # The w3j nodes are reshapes, so we have to remove them from the graph
            # Although they are dead code, they try to reshape to dimensions that don't exist
            # (since the corresponding w3js are not in w3j)
            # so they screw up the shape propagation, even though they would be removed later as dead code by TorchScript.
            graph.erase_node(w3j.node)
        else:
            if w3j_name not in constants:
                constants[w3j_name] = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

    # = Return the result =
    outputs = [
        _sum_tensors(
            [out for ins, out in zip(instructions, outputs) if ins.i_out == i_out],
            shape=(batch_numel, mul_ir_out.dim),
            like=x1s
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(outputs) > 1:
        outputs = torch.cat(outputs, dim=1)
    else:
        # Avoid an unnecessary copy in a size one torch.cat
        outputs = outputs[0]

    outputs = outputs.reshape(output_shape)

    graph.output(outputs.node, torch.Tensor)

    # check graphs
    graph.lint()

    # Make GraphModules

    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    for key, value in constants.items():
        constants_root.register_buffer(key, value)
    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_forward")

    # == Optimize ==
    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
        # and still get the right path for _all_ batch dimensions.
        # This is because our einsums are essentially of the form:
        #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
        # In the first case, all but one operands have the batch dimension
        #    => The first contraction gains the batch dimension
        #    => All following contractions have batch dimension
        #    => All possible contraction paths have cost that scales linearly in batch size
        #    => The optimal path is the same for all batch sizes
        # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
        # https://github.com/dgasmith/opt_einsum/issues/158
        # for more details.
        #
        # TODO: consider the impact maximum intermediate result size on this logic
        #         \- this is the `memory_limit` option in opt_einsum
        # TODO: allow user to choose opt_einsum parameters?
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, irreps_in1.dim)),
            torch.zeros((batchdim, irreps_in2.dim)),
            torch.zeros(
                1 if shared_weights else batchdim,
                flat_weight_index,
            ),
        )
        graphmod = optimize_einsums_full(graphmod, example_inputs)

    return graphmod













@compile_mode('script')
class TensorProduct(CodeGenMixin, torch.nn.Module):
    def __init__(self, n_s):

        super().__init__()

        # good stuff
        self.irreps_in1 = o3.Irreps(f"{n_s}x0e")
        self.irreps_in2 = o3.Irreps(o3.Irreps.spherical_harmonics(lmax=0))
        self.irreps_out = o3.Irreps(f"{n_s}x0e")
        self.instructions = [
            Instruction(
                i_in1=0,
                i_in2=0,
                i_out=0,
                connection_mode='uvw',
                has_weight=True,
                path_weight=sqrt(1.0/n_s),
                path_shape= (n_s, 1, n_s),
            )
        ]
        self._in1_dim = n_s
        self._in2_dim = 1
        self.internal_weights = False
        self.shared_weights = False

        # ??? generate the actual tensor product code
        graphmod_left_right = codegen_tensor_product_left_right(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            self.shared_weights,
            True, True
        )
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

        # Weights
        self.weight_numel = n_s * n_s
        self.register_buffer('weight', torch.Tensor())

        # ??? stuff
        output_mask = torch.ones(n_s)
        self.register_buffer('output_mask', output_mask)
        self._profiling_str = str(self)


    def __repr__(self):
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} "
            f"-> {self.irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)"
        )

    def forward(self, x, y, weight):
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"
        assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
        return self._compiled_main_left_right(x, y, weight)
