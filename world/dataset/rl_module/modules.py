# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from https://github.com/pytorch/rl/blob/main/torchrl

from typing import Optional, Sequence, Type, Union

import torch
from torch import nn

from .utils import (
    _find_depth,
    create_on_device,
    LazyMapping,
    SquashDims,
    Squeeze2dLayer
)
from .device import DEVICE_TYPING

class ConvNet(nn.Sequential):
    def __init__(
        self,
        in_features: Optional[int] = None,
        depth: Optional[int] = None,
        num_cells: Union[Sequence, int] = None,
        kernel_sizes: Union[Sequence[Union[int, Sequence[int]]], int] = 3,
        strides: Union[Sequence, int] = 1,
        paddings: Union[Sequence, int] = 0,
        activation_class: Type[nn.Module] = nn.ELU,
        activation_kwargs: Optional[dict] = None,
        norm_class: Optional[Type[nn.Module]] = None,
        norm_kwargs: Optional[dict] = None,
        bias_last_layer: bool = True,
        aggregator_class: Optional[Type[nn.Module]] = SquashDims,
        aggregator_kwargs: Optional[dict] = None,
        squeeze_output: bool = False,
        device: Optional[DEVICE_TYPING] = None,
    ):
        if num_cells is None:
            num_cells = [32, 32, 32]

        self.in_features = in_features
        self.activation_class = activation_class
        self.activation_kwargs = (
            activation_kwargs if activation_kwargs is not None else {}
        )
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.bias_last_layer = bias_last_layer
        self.aggregator_class = aggregator_class
        self.aggregator_kwargs = (
            aggregator_kwargs if aggregator_kwargs is not None else {"ndims_in": 3}
        )
        self.squeeze_output = squeeze_output
        # self.single_bias_last_layer = single_bias_last_layer

        depth = _find_depth(depth, num_cells, kernel_sizes, strides, paddings)
        self.depth = depth
        if depth == 0:
            raise ValueError("Null depth is not permitted with ConvNet.")

        for _field, _value in zip(
            ["num_cells", "kernel_sizes", "strides", "paddings"],
            [num_cells, kernel_sizes, strides, paddings],
        ):
            _depth = depth
            setattr(
                self,
                _field,
                (_value if isinstance(_value, Sequence) else [_value] * _depth),
            )
            if not (isinstance(_value, Sequence) or _depth is not None):
                raise RuntimeError(
                    f"If {_field} is provided as an integer, "
                    "depth must be provided too."
                )
            if not (len(getattr(self, _field)) == _depth or _depth is None):
                raise RuntimeError(
                    f"depth={depth} and {_field}={len(getattr(self, _field))} length conflict, "
                    + f"consider matching or specifying a constant {_field} argument together with a a desired depth"
                )

        self.out_features = self.num_cells[-1]

        self.depth = len(self.kernel_sizes)
        layers = self._make_net(device)
        super().__init__(*layers)

    def _make_net(self, device: Optional[DEVICE_TYPING]) -> nn.Module:
        layers = []
        in_features = [self.in_features] + self.num_cells[: self.depth]
        out_features = self.num_cells + [self.out_features]
        kernel_sizes = self.kernel_sizes
        strides = self.strides
        paddings = self.paddings
        for i, (_in, _out, _kernel, _stride, _padding) in enumerate(
            zip(in_features, out_features, kernel_sizes, strides, paddings)
        ):
            _bias = (i < len(in_features) - 1) or self.bias_last_layer
            if _in is not None:
                layers.append(
                    nn.Conv2d(
                        _in,
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=device,
                    )
                )
            else:
                layers.append(
                    nn.LazyConv2d(
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=device,
                    )
                )

            layers.append(
                create_on_device(
                    self.activation_class, device, **self.activation_kwargs
                )
            )
            if self.norm_class is not None:
                layers.append(
                    create_on_device(self.norm_class, device, **self.norm_kwargs)
                )

        if self.aggregator_class is not None:
            layers.append(
                create_on_device(
                    self.aggregator_class, device, **self.aggregator_kwargs
                )
            )

        if self.squeeze_output:
            layers.append(Squeeze2dLayer())
        return layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, C, L, W = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = super(ConvNet, self).forward(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


Conv2dNet = ConvNet

class MultiAgentConvNet(nn.Module):
    def __init__(
        self,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: Optional[DEVICE_TYPING] = None,
        num_cells: Optional[Sequence[int]] = None,
        kernel_sizes: Union[Sequence[Union[int, Sequence[int]]], int] = 5,
        strides: Union[Sequence, int] = 2,
        paddings: Union[Sequence, int] = 0,
        activation_class: Type[nn.Module] = nn.ELU,
        **kwargs,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.centralised = centralised
        self.share_params = share_params

        self.agent_networks = nn.ModuleList(
            [
                ConvNet(
                    num_cells=num_cells,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    paddings=paddings,
                    activation_class=activation_class,
                    device=device,
                    **kwargs,
                )
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

    def forward(self, inputs: torch.Tensor):
        if len(inputs.shape) < 4:
            raise ValueError(
                """Multi-agent network expects (*batch_size, agent_index, x, y, channels)"""
            )
        if inputs.shape[-4] != self.n_agents:
            raise ValueError(
                f"""Multi-agent network expects {self.n_agents} but got {inputs.shape[-4]}"""
            )
        # If the model is centralized, agents have full observability
        if self.centralised:
            shape = (
                *inputs.shape[:-4],
                self.n_agents * inputs.shape[-3],
                inputs.shape[-2],
                inputs.shape[-1],
            )
            inputs = torch.reshape(inputs, shape)

        # If the parameters are not shared, each agent has its own network
        if not self.share_params:
            if self.centralised:
                output = torch.stack(
                    [net(inputs) for net in self.agent_networks], dim=-2
                )
            else:
                output = torch.stack(
                    [
                        net(inp)
                        for i, (net, inp) in enumerate(
                            zip(self.agent_networks, inputs.unbind(-4))
                        )
                    ],
                    dim=-2,
                )
        else:
            output = self.agent_networks[0](inputs)
            if self.centralised:
                # If the parameters are shared, and it is centralised all agents will have the same output.
                # We expand it to maintain the agent dimension, but values will be the same for all agents
                n_agent_outputs = output.shape[-1]
                output = output.view(*output.shape[:-1], n_agent_outputs)
                output = output.unsqueeze(-2)
                output = output.expand(
                    *output.shape[:-2], self.n_agents, n_agent_outputs
                )
        return output
