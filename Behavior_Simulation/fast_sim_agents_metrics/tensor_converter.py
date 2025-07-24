import functools
import tensorflow as tf
import torch
from typing import Any, Dict, List, Union, TypeVar, Callable

T = TypeVar('T')

def _convert_tf_to_torch(tensor: tf.Tensor) -> torch.Tensor:
    """将 TensorFlow 张量转换为 PyTorch 张量"""
    if not isinstance(tensor, tf.Tensor):
        return tensor
    
    # 获取张量的形状和数据类型
    shape = tensor.shape.as_list()
    dtype = tensor.dtype.as_numpy_dtype
    
    # 将 TensorFlow 张量转换为 numpy 数组
    numpy_array = tensor.numpy()
    
    # 创建对应的 PyTorch 张量
    torch_dtype = {
        tf.float32: torch.float32,
        tf.float64: torch.float64,
        tf.int32: torch.int32,
        tf.int64: torch.int64,
        tf.bool: torch.bool,
    }.get(dtype, torch.float32)
    
    return torch.from_numpy(numpy_array).to(torch_dtype)

def _convert_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """递归转换字典中的所有 TensorFlow 张量"""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _convert_dict(value)
        elif isinstance(value, (list, tuple)):
            result[key] = _convert_list(value)
        elif isinstance(value, tf.Tensor):
            result[key] = _convert_tf_to_torch(value)
        else:
            result[key] = value
    return result

def _convert_list(l: Union[List[Any], tuple]) -> Union[List[Any], tuple]:
    """递归转换列表或元组中的所有 TensorFlow 张量"""
    result = []
    for item in l:
        if isinstance(item, dict):
            result.append(_convert_dict(item))
        elif isinstance(item, (list, tuple)):
            result.append(_convert_list(item))
        elif isinstance(item, tf.Tensor):
            result.append(_convert_tf_to_torch(item))
        else:
            result.append(item)
    return type(l)(result)

def _convert_class_instance(obj: Any) -> Any:
    """递归转换类实例中的所有 TensorFlow 张量"""
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if isinstance(value, dict):
                setattr(obj, key, _convert_dict(value))
            elif isinstance(value, (list, tuple)):
                setattr(obj, key, _convert_list(value))
            elif isinstance(value, tf.Tensor):
                setattr(obj, key, _convert_tf_to_torch(value))
            elif hasattr(value, '__dict__'):
                setattr(obj, key, _convert_class_instance(value))
    return obj

def convert_tf_to_torch(func: Callable[..., T]) -> Callable[..., T]:
    """装饰器：将函数输入中的 TensorFlow 张量转换为 PyTorch 张量"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 转换位置参数
        converted_args = []
        for arg in args:
            if isinstance(arg, dict):
                converted_args.append(_convert_dict(arg))
            elif isinstance(arg, (list, tuple)):
                converted_args.append(_convert_list(arg))
            elif isinstance(arg, tf.Tensor):
                converted_args.append(_convert_tf_to_torch(arg))
            elif hasattr(arg, '__dict__'):
                converted_args.append(_convert_class_instance(arg))
            else:
                converted_args.append(arg)
        
        # 转换关键字参数
        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                converted_kwargs[key] = _convert_dict(value)
            elif isinstance(value, (list, tuple)):
                converted_kwargs[key] = _convert_list(value)
            elif isinstance(value, tf.Tensor):
                converted_kwargs[key] = _convert_tf_to_torch(value)
            elif hasattr(value, '__dict__'):
                converted_kwargs[key] = _convert_class_instance(value)
            else:
                converted_kwargs[key] = value
        
        return func(*converted_args, **converted_kwargs)
    return wrapper 