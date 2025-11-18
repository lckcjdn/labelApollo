import torch
from segment_anything import sam_model_registry
from inspect import signature

# 加载模型配置
model_type = "vit_b"
checkpoint_path = "d:\\Ar\\labelApollo\\backend\\models\\sam_vit_b.pth"

# 初始化模型但不加载权重（这样更快）
sam = sam_model_registry[model_type]()

# 检查MaskDecoder的forward方法参数
mask_decoder = sam.mask_decoder
forward_method = mask_decoder.forward
params = list(signature(forward_method).parameters.keys())

print("MaskDecoder forward方法参数:")
for param in params:
    print(f"  - {param}")

print("\nMaskDecoder类结构:")
print(dir(mask_decoder))

# 检查prompt_encoder的返回值类型
prompt_encoder = sam.prompt_encoder
print("\nPromptEncoder类结构:")
print(dir(prompt_encoder))

# 检查get_dense_pe方法
if hasattr(prompt_encoder, 'get_dense_pe'):
    print("\nget_dense_pe方法存在")
else:
    print("\nget_dense_pe方法不存在")