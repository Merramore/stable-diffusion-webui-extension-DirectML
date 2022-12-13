#!python3
#hacks.py

from sys import stderr
from functools import wraps
from importlib import import_module

import torch

from modules import shared, devices


try:
    import torch_directml as dml
except ModuleNotFoundError:
    has_dml = False
else:
    has_dml = True



def inject_func(modpath, objpath, func):
    mod = import_module(modpath)
    objnames = objpath.split('.')

    container = mod
    for objname in objnames[:-1]:
        container = getattr(container, objname)
    objname = objnames[-1]

    setattr(container, objname, func)

    # print(f"[Shims] Injected {modpath}:{objpath}.", file=stderr)
    print(f"[DirectML] Injected {modpath}:{objpath}.")
    return func

def injectable_func(modpath, objpath):
    def impl (func):
        return inject_func(modpath, objpath, func)
    return impl


def inject_shim(modpath, objpath, shim):
    mod = import_module(modpath)
    objnames = objpath.split('.')

    container = None
    injection_target = mod
    for objname in objnames:
        container = injection_target
        injection_target = getattr(container, objname)

    @wraps(injection_target)
    def injection_wrapper(*args, **kwargs):
        return shim(injection_target, *args, **kwargs)
    setattr(container, objname, injection_wrapper)

    # print(f"[Smhims] Injected {modpath}:{objpath}.", file=stderr)
    print(f"[DirectML] Injected {modpath}:{objpath}.")
    return injection_wrapper

def injectable_shim(modpath, objpath):
    def impl(shim):
        return inject_shim(modpath, objpath, shim)
    return impl



# == torch hacks == #

if has_dml:
    dml_devices = [dml.device(_i) for _i in range(dml.device_count())]

    # Bypass unimplemented ops: torch.group_norm(), torch.Tensor.cumsum
    def bypass_dml_shim(shim_target, *args, **kwargs):
        dml_device = None
        def maybe_bypass_arg(arg):
            nonlocal dml_device
            try:
                device = arg.device
            except:
                pass
            else:
                if device in dml_devices:
                    dml_device = device
                    return arg.to('cpu')
            return arg
        output = shim_target(
            *(maybe_bypass_arg(arg) for arg in args),
            **{k:maybe_bypass_arg(v) for k, v in kwargs.items()},
        )
        if dml_device is not None:
            # Send it back from where it came, or else downstream ops will complain about mixed devices.
            output = output.to(dml_device)
        return output

    group_norm = inject_shim("torch", "group_norm", bypass_dml_shim)
    cumsum = inject_shim("torch", "Tensor.cumsum", bypass_dml_shim)


    @injectable_shim("torch", "Tensor.new")
    def new_tensor(shim_target, self, *args, **kwargs):
        device = self.device
        if device.type == 'privateuseone':
            return shim_target(self.to('cpu'), *args, **kwargs).to(device)
        return shim_target(self, *args, **kwargs)



# == webui hacks == #

@injectable_func("modules.devices", "get_device")
def get_device(device_string):
    if has_dml and device_string.lower().startswith("dml"):
        device_id = device_string[3:].strip(":")
        device_id = int(device_id) if device_id else None
        return dml.device(device_id)

    return torch.device(device_string)


@injectable_shim("modules.devices", "get_optimal_device")
def get_optimal_device(original_func):
    if shared.cmd_opts.device is not None:
        return get_device(shared.cmd_opts.device)

    if torch.cuda.is_available():
        return torch.device(devices.get_cuda_device_string())

    if has_dml:
        return dml.device()

    return original_func()
