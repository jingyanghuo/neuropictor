import os, cv2, torch, math, pickle, torch, copy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.io import loadmat
import tensorflow.compat.v1 as tf
import msgpack, enum



def _dtype_from_name(name: str):
  """Handle JAX bfloat16 dtype correctly."""
  # if name == b'bfloat16':
  #   return jax.numpy.bfloat16
  # else:
  #   return np.dtype(name)
  return np.dtype(name)

class _MsgpackExtType(enum.IntEnum):
  """Messagepack custom type ids."""
  ndarray = 1
  native_complex = 2
  npscalar = 3

def _ndarray_from_bytes(data: bytes) -> np.ndarray:
  """Load ndarray from simple msgpack encoding."""
  shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
  return np.frombuffer(buffer,
                       dtype=_dtype_from_name(dtype_name),
                       count=-1,
                       offset=0).reshape(shape, order='C')

def _msgpack_ext_unpack(code, data):
  """Messagepack decoders for custom types."""
  if code == _MsgpackExtType.ndarray:
    return _ndarray_from_bytes(data)
  elif code == _MsgpackExtType.native_complex:
    complex_tuple = msgpack.unpackb(data)
    return complex(complex_tuple[0], complex_tuple[1])
  elif code == _MsgpackExtType.npscalar:
    ar = _ndarray_from_bytes(data)
    return ar[()]  # unpack ndarray to scalar
  return msgpack.ExtType(code, data)

def load_state(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        # file = f.read()
        state = msgpack.unpack(f, ext_hook=_msgpack_ext_unpack, raw=False)

        return state

def tokenizer_to_pytorch_ckpt(path):

    layer_key_map = {
        'GroupNorm_0': 'norm1',
        'Conv_0': 'conv1',
        'GroupNorm_1':  'norm2',
        'Conv_1': 'conv2',
        'Conv_2': 'nin_shortcut',
    }

    def recursive_check(prefix, state, output):
        if isinstance(state, dict):
            for k, v in state.items():
                recursive_check('%s.%s' % (prefix, k), state[k], output)
        else:
            value = torch.from_numpy(state.copy())
            if value.dim() == 4:
                value = value.permute(3,2,0,1).contiguous() # [H, W, in, out] -> [out, in, H, W]
            output[prefix] = value

    def recursive_encoder(prefix, state, output):
        if isinstance(state, dict):
            for k, v in state.items():
                recursive_encoder('%s.%s' % (prefix, k), state[k], output)
        else:
            # specific naming rules for pytorch model
            prefix = prefix.replace('kernel', 'weight')
            prefix = prefix.replace('scale', 'weight')
            prefix = prefix.replace('encoder.Conv_0', 'encoder.conv_in')
            prefix = prefix.replace('encoder.Conv_1', 'encoder.conv_out')
            prefix = prefix.replace('encoder.GroupNorm_0', 'encoder.norm_out')

            if 'ResBlock' in prefix:
                parts = prefix.split('.')
                res = parts[1]
                layer = parts[2]
                count = int(res.split('_')[-1])

                if count < 10:
                    n_block = str(count // 2)
                    n_res = str(count % 2)

                    prefix = '%s.%s.%s.%s.%s.%s.%s' % (parts[0], 'down', n_block, 'block', n_res, layer_key_map[layer], parts[-1])
                else:
                    n_block = str(count % 10)
                    prefix = '%s.%s.%s.%s.%s' % (parts[0], 'mid', n_block, layer_key_map[layer], parts[-1])

            value = torch.from_numpy(state.copy())
            if value.dim() == 4:
                value = value.permute(3,2,0,1).contiguous() # [H, W, in, out] -> [out, in, H, W]
            output[prefix] = value

    def recursive_decoder(prefix, state, output):
        if isinstance(state, dict):
            for k, v in state.items():
                recursive_decoder('%s.%s' % (prefix, k), state[k], output)
        else:
            # speficic naming rules for pytorch model
            prefix = prefix.replace('kernel', 'weight')
            prefix = prefix.replace('scale', 'weight')
            prefix = prefix.replace('decoder.Conv_0', 'decoder.conv_in')
            prefix = prefix.replace('decoder.Conv_5', 'decoder.conv_out')
            prefix = prefix.replace('decoder.Conv_1', 'decoder.up.0.upsample.conv')
            prefix = prefix.replace('decoder.Conv_2', 'decoder.up.1.upsample.conv')
            prefix = prefix.replace('decoder.Conv_3', 'decoder.up.2.upsample.conv')
            prefix = prefix.replace('decoder.Conv_4', 'decoder.up.3.upsample.conv')
            prefix = prefix.replace('decoder.GroupNorm_0', 'decoder.norm_out')

            if 'ResBlock' in prefix:
                parts = prefix.split('.')
                res = parts[1]
                layer = parts[2]
                count = int(res.split('_')[-1])

                if count > 1:
                    n_block = str(count // 2 - 1)
                    n_res = str(count % 2)

                    prefix = '%s.%s.%s.%s.%s.%s.%s' % (parts[0], 'up', n_block, 'block', n_res, layer_key_map[layer], parts[-1])
                else:
                    n_block = str(count)
                    prefix = '%s.%s.%s.%s.%s' % (parts[0], 'mid', n_block, layer_key_map[layer], parts[-1])

            value = torch.from_numpy(state.copy())
            if value.dim() == 4:
                value = value.permute(3,2,0,1).contiguous() # [H, W, in, out] -> [out, in, H, W]
            output[prefix] = value

    with tf.io.gfile.GFile(path, 'rb') as f:
        # file = f.read()
        state = msgpack.unpack(f, ext_hook=_msgpack_ext_unpack, raw=False)

        # encoder
        encoder_state = {}
        recursive_encoder('encoder', state['params']['encoder'], encoder_state)

        # decoder
        decoder_state = {}
        recursive_decoder('decoder', state['params']['decoder'], decoder_state)

        # quantize
        quantizer_state = {'quantizer.embedding.weight': torch.from_numpy(state['params']['quantizer']['codebook'].copy())}

        state = {}
        state.update(encoder_state)
        state.update(decoder_state)
        state.update(quantizer_state)

        # print (len(encoder_state.keys()), encoder_state.keys())
        # print (len(decoder_state.keys()), decoder_state.keys())
        # print (len(quantizer_state.keys()), quantizer_state.keys())
        return state

def transformer_to_pytorch_ckpt(path):

    def recursive_check(prefix, state, output):
        if isinstance(state, dict):
            for k, v in state.items():
                recursive_check('%s.%s' % (prefix, k), state[k], output)
        else:
            value = torch.from_numpy(state.copy())
            if value.dim() == 4:
                value = value.permute(3,2,0,1).contiguous() # [H, W, in, out] -> [out, in, H, W]
            print (prefix, value.shape)
            output[prefix] = value

    def recursive_embedding(prefix, state, output):
        if isinstance(state, dict):
            for k, v in state.items():
                recursive_embedding('%s.%s' % (prefix, k), state[k], output)
        else:

            prefix = prefix.replace('embeddings_ln', 'ln')
            prefix = prefix.replace('position_embeddings', 'de_pos_emb')
            prefix = prefix.replace('word_embeddings', 'token_emb')
            prefix = prefix.replace('.embedding', '.weight')
            prefix = prefix.replace('scale', 'weight')

            value = torch.from_numpy(state.copy())
            output[prefix] = value

    def recursive_transformer(prefix, state, output):
        if isinstance(state, dict):
            for k, v in state.items():
                recursive_transformer('%s.%s' % (prefix, k), state[k], output)
        else:
            prefix = prefix.replace('TransformerLayer_', 'decoder.')
            prefix = prefix.replace('Attention_0.', '')
            prefix = prefix.replace('self_attention', 'self_attn')
            prefix = prefix.replace('kernel', 'weight')
            prefix = prefix.replace('scale', 'weight')
            prefix = prefix.replace('attention_output_ln', 'norm1')
            prefix = prefix.replace('Mlp_0.layer_output_ln', 'norm2')
            prefix = prefix.replace('Mlp_0.intermediate_output', 'linear1')
            prefix = prefix.replace('Mlp_0.layer_output', 'linear2')
            prefix = prefix.replace('out', 'proj')

            value = torch.from_numpy(state.copy())
            if 'key' in prefix or 'query' in prefix or 'value' in prefix:
                value = value.flatten(start_dim=-2)
                if 'weight' in prefix:
                    value = value.transpose(0,1).contiguous() # [L, hC]
            if 'proj.weight' in prefix:
                value = value.flatten(end_dim=1).transpose(0,1).contiguous() # [hC, L]
            if 'linear' in prefix and 'weight' in prefix:
                value = value.transpose(0,1).contiguous() # [in, out] -> [out, in]
            output[prefix] = value

    def recursive_mlps(prefix, state, output):
        if isinstance(state, dict):
            for k, v in state.items():
                recursive_mlps('%s.%s' % (prefix, k), state[k], output)
        else:
            prefix = prefix.replace('mlm_dense', 'mlps.0')
            prefix = prefix.replace('mlm_ln', 'mlps.2')
            prefix = prefix.replace('mlm_bias.bias', 'mlps_bias')
            prefix = prefix.replace('kernel', 'weight')
            prefix = prefix.replace('scale', 'weight')

            value = torch.from_numpy(state.copy())
            if 'weight' in prefix and value.dim() == 2:
                value = value.transpose(0,1).contiguous() # [in, out] -> [out, in]
            output[prefix] = value

    with tf.io.gfile.GFile(path, 'rb') as f:
        # file = f.read()
        state = msgpack.unpack(f, ext_hook=_msgpack_ext_unpack, raw=False)

        # Embedding
        embedding_state = {}
        recursive_embedding('transformer', state['params']['Embed_0'], embedding_state)


        # transformer
        trans_state = {}
        recursive_transformer('transformer', {k: v for k, v in state['params'].items() if 'TransformerLayer' in k}, trans_state)

        # mlps
        mlps_state = {}
        recursive_mlps('transformer', state['params']['MlmLayer_0'], mlps_state)

        state = {}
        state.update(embedding_state)
        state.update(trans_state)
        state.update(mlps_state)

        # print (len(state.keys()))
        # print (state.keys())

        return state









if __name__ == '__main__':
    '''
    path = '/home/WORKSPACE/workspace/qxl/mind-vis-main/VQ-branch/pretrains/tokenizer_imagenet256_checkpoint'
    state = tokenizer_to_pytorch_ckpt(path)

    torch.save(state,'/home/WORKSPACE/workspace/qxl/mind-vis-main/VQ-branch/pretrains/MaskGIT_ImageNet256_checkpoint.pth')
    print (len(state.keys()))
    # vq = state['params']['quantizer']['codebook']
    # print (vq.shape)
    '''

    path = '/home/WORKSPACE/workspace/qxl/mind-vis-main/VQ/pretrains/maskgit_imagenet256_checkpoint'
    state = transformer_to_pytorch_ckpt(path)
    torch.save(state,'/home/WORKSPACE/workspace/qxl/mind-vis-main/VQ/pretrains/MaskGIT_Trans_ImageNet256_checkpoint.pth')
    print (len(state.keys()))