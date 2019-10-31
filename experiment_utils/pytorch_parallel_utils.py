import torch
from itertools import chain
from torch.nn.parallel import (gather,
                               replicate,
                               scatter_kwargs,
                               parallel_apply)
from torch.cuda._utils import _get_device_index


def data_parallel_with_post_processing(
        func,
        module,
        inputs,
        device_ids=None,
        output_device=None,
        dim=0,
        module_kwargs=None):

    r"""This is the functional version of the DataParallel module,
        with additional support for processing outputs *before*
        gathering.
    """
    if not callable(func):
        raise TypeError("`func` has to be callable, "
                        "but found {}".format(type(func)))

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device("cuda:{}".format(device_ids[0]))

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError(
                "module must have its parameters and buffers "
                "on device {} (device_ids[0]) but found one of "
                "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(
        inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)

    # Process Outputs Before Gathering
    outputs = func(outputs)
    return gather(outputs, output_device, dim)
