import numpy as np
import random
import torch
from functools import wraps
import torch.utils._device
import time


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None):
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, '__module__', None) == 'torch.nn.init':
            if 'tensor' in kwargs:
                return kwargs['tensor']
            else:
                return args[0]
        if self.device is not None and func in torch.utils._device._device_constructors() and kwargs.get('device') is None:
            kwargs['device'] = self.device
        return func(*args, **kwargs)
    

def with_empty_init(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with EmptyInitOnDevice('cpu'):
            return func(*args, **kwargs)
    return wrapper


    
    
def culens2mask(
    cu_seqlens=None,
    cu_seqlens_kv=None,
    max_seqlen=None,
    max_seqlen_kv=None,
    is_causal=False
):
    assert len(cu_seqlens) == len(cu_seqlens_kv); "q k v should have same bsz..."
    bsz = len(cu_seqlens) - 1
    seqlens = cu_seqlens[1:]-cu_seqlens[:-1]
    seqlens_kv = cu_seqlens_kv[1:]-cu_seqlens_kv[:-1]
    
    attn_mask = torch.zeros(bsz, max_seqlen, max_seqlen_kv, dtype=torch.bool)
    for i, (seq_len, seq_len_kv) in enumerate(zip(seqlens, seqlens_kv)):
        if is_causal:
            attn_mask[i, :seq_len, :seq_len_kv] = torch.triu(torch.ones(seq_len, seq_len_kv), diagonal=1).bool()
        else:
            attn_mask[i, :seq_len, :seq_len_kv] = torch.ones([seq_len, seq_len_kv], dtype=torch.bool)

    return attn_mask


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            print(f"{class_name}.{func.__name__} took {elapsed_time:.3f} seconds")
        else:
            print(f"{func.__name__} took {elapsed_time:.3f} seconds")
        return result
    return wrapper