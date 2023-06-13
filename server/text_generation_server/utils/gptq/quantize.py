import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import math
import json
import os

from texttable import Texttable
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import transformers
import numpy as np
import torch
from text_generation_server.utils.gptq.quant_linear import QuantLinear
from loguru import logger

DEV = torch.device("cuda:0")


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=.8, trits=False):

        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.scale = torch.zeros_like(self.scale)

    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = self._quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return self._quantize(x, self.scale, self.zero, self.maxq)

        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, layer, observe=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.observe = observe

    def add_batch(self, inp, out):
        # Hessian H = 2 X XT + λ I
        if self.observe:
            self.inp1 = inp
            self.out1 = out
        else:
            self.inp1 = None
            self.out1 = None

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def print_loss(self, name, q_weight, weight_error, timecost):
        table = Texttable()
        name += ' ' * (16 - len(name))

        table.header(['name', 'weight_error', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.inp1 is not None:
            # quantize input to int8
            quantizer = Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'

        table.add_row([name, weight_error, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name=''):
        self.layer.to(self.dev)

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        if not self.observe:
            del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.print_loss(name=name, q_weight=Q, weight_error=error, timecost=(time.time() - tick))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, error

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


def get_wikitext2(nsamples, seed, seqlen, model_id):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model_id):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model_id):
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False)
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', use_auth_token=False)

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model_id):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model_id):
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model_id=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model_id)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model_id)
        return get_ptb(nsamples, seed, seqlen, model_id)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model_id)
        return get_c4(nsamples, seed, seqlen, model_id)


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    # Skip last lm_head linear
    if type(module) in layers and "lm_head" not in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

@torch.no_grad()
def sequential(model, dataloader, dev, nsamples, bits, groupsize, percdamp=0.01, sym: bool=False, act_order: bool = False):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # embeddings = model.get_input_embeddings()
    # embeddings = embeddings.to(dev)
    # model.set_input_embeddings(embeddings)
    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    # layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module

    # layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask'].to(dev)
    position_ids = cache['position_ids'].to(dev)

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        from accelerate.hooks import remove_hook_from_submodules
        layer = layers[i].to(dev)
        remove_hook_from_submodules(layer)
        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer.configure(bits, perchannel=True, sym=sym, mse=False)

            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):

                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=percdamp, groupsize=groupsize, actorder=act_order, name=name)
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), bits, groupsize)

                gptq[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    # if args.observe:
    #     observer.print()
    #     conditions = gen_conditions(args.bits, args.groupsize)
    #     for item in observer.items():
    #         name = item[0]
    #         layerid = item[1]
    #         gptq = item[2]['gptq']
    #         error = item[2]['error']
    #         target = error / 2

    #         table = Texttable()
    #         table.header(['bits', 'groupsize', 'error'])
    #         table.set_cols_dtype(['i', 'i', 'f'])
    #         table.add_row([args.bits, args.groupsize, error])

    #         print('Optimizing {} {} ..'.format(name, layerid))
    #         for bits, groupsize in conditions:

    #             if error < target:
    #                 # if error dropped 50%, skip
    #                 break

    #             gptq.quantizer.configure(bits, perchannel=True, sym=args.sym, mse=False)

    #             scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

    #             table.add_row([bits, groupsize, error])
    #             quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), bits, groupsize)

    #         print(table.draw())
    #         print('\n')
    #         gptq.layer.to('cpu')
    #         gptq.free()

    model.config.use_cache = use_cache

    return quantizers


# @torch.no_grad()
# def llama_eval(model, testenc, dev):
#     print('Evaluating ...')
# 
#     testenc = testenc.input_ids
#     nsamples = testenc.numel() // model.seqlen
# 
#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = model.model.layers
# 
#     model.model.embed_tokens = model.model.embed_tokens.to(dev)
#     layers[0] = layers[0].to(dev)
# 
#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
#     cache = {'i': 0, 'attention_mask': None}
# 
#     class Catcher(nn.Module):
# 
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
# 
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             cache['attention_mask'] = kwargs['attention_mask']
#             cache['position_ids'] = kwargs['position_ids']
#             raise ValueError
# 
#     layers[0] = Catcher(layers[0])
#     for i in range(nsamples):
#         batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
#         try:
#             model(batch)
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
# 
#     layers[0] = layers[0].cpu()
#     model.model.embed_tokens = model.model.embed_tokens.cpu()
#     torch.cuda.empty_cache()
# 
#     outs = torch.zeros_like(inps)
#     attention_mask = cache['attention_mask']
#     position_ids = cache['position_ids']
# 
#     for i in range(len(layers)):
#         print(i)
#         layer = layers[i].to(dev)
# 
#         if args.nearest:
#             subset = find_layers(layer)
#             for name in subset:
#                 quantizer = quant.Quantizer()
#                 quantizer.configure(args.bits, perchannel=True, sym=args.sym, mse=False)
#                 W = subset[name].weight.data
#                 quantizer.find_params(W, weight=True)
#                 subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)
# 
#         for j in range(nsamples):
#             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#         layers[i] = layer.cpu()
#         del layer
#         torch.cuda.empty_cache()
#         inps, outs = outs, inps
# 
#     if model.model.norm is not None:
#         model.model.norm = model.model.norm.to(dev)
#     model.lm_head = model.lm_head.to(dev)
# 
#     testenc = testenc.to(dev)
#     nlls = []
#     for i in range(nsamples):
#         hidden_states = inps[i].unsqueeze(0)
#         if model.model.norm is not None:
#             hidden_states = model.model.norm(hidden_states)
#         lm_logits = model.lm_head(hidden_states)
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#         neg_log_likelihood = loss.float() * model.seqlen
#         nlls.append(neg_log_likelihood)
#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
#     print(ppl.item())
# 
#     model.config.use_cache = use_cache

def make_quant_linear(module, names, bits, groupsize, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear.new(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)



# TODO: perform packing on GPU
def pack(model, quantizers, bits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant_linear(model, quantizers, bits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


# def load_quant(model, checkpoint, bits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
#     from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
#     config = LlamaConfig.from_pretrained(model)
# 
#     def noop(*args, **kwargs):
#         pass
# 
#     torch.nn.init.kaiming_uniform_ = noop
#     torch.nn.init.uniform_ = noop
#     torch.nn.init.normal_ = noop
# 
#     torch.set_default_dtype(torch.half)
#     modeling_utils._init_weights = False
#     torch.set_default_dtype(torch.half)
#     model = LlamaForCausalLM(config)
#     torch.set_default_dtype(torch.float)
#     if eval:
#         model = model.eval()
#     layers = find_layers(model)
#     for name in ['lm_head']:
#         if name in layers:
#             del layers[name]
#     quant.make_quant_linear(model, layers, bits, groupsize)
# 
#     del layers
# 
#     print('Loading model ...')
#     if checkpoint.endswith('.safetensors'):
#         from safetensors.torch import load_file as safe_load
#         model.load_state_dict(safe_load(checkpoint))
#     else:
#         model.load_state_dict(torch.load(checkpoint))
# 
#     if eval:
#         quant.make_quant_attn(model)
#         quant.make_quant_norm(model)
#         if fused_mlp:
#             quant.make_fused_mlp(model)
# 
#     if warmup_autotune:
#         quant.autotune_warmup_linear(model, transpose=not (eval))
#         if eval and fused_mlp:
#             quant.autotune_warmup_fused(model)
#     model.seqlen = 2048
#     print('Done.')
# 
#     return model


# def llama_multigpu(model, gpus, gpu_dist):
#     model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
#     if hasattr(model.model, 'norm') and model.model.norm:
#         model.model.norm = model.model.norm.to(gpus[0])
#     import copy
#     model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])
# 
#     cache = {'mask': None, 'position_ids': None}
# 
#     class MoveModule(nn.Module):
# 
#         def __init__(self, module, invalidate_cache):
#             super().__init__()
#             self.module = module
#             self.dev = next(iter(self.module.parameters())).device
#             self.invalidate_cache=invalidate_cache
# 
#         def forward(self, *inp, **kwargs):
#             inp = list(inp)
#             if inp[0].device != self.dev:
#                 inp[0] = inp[0].to(self.dev)
# 
#             if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
#                 cache['mask'] = kwargs['attention_mask'].to(self.dev)
#             kwargs['attention_mask'] = cache['mask']
# 
#             if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
#                 cache['position_ids'] = kwargs['position_ids'].to(self.dev)
#             kwargs['position_ids'] = cache['position_ids']
#             
#             tmp = self.module(*inp, **kwargs)
#             return tmp
# 
#     layers = model.model.layers
#     from math import ceil
#     if not gpu_dist:
#         pergpu = ceil(len(layers) / len(gpus))
#         for i in range(len(layers)):
#             layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
#     else:
#         assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
#         assigned_gpus = [0] * (gpu_dist[0]-1)
#         for i in range(1, len(gpu_dist)):
#             assigned_gpus = assigned_gpus + [i] * gpu_dist[i]
# 
#         remaining_assignments = len(layers)-len(assigned_gpus) - 1
#         if remaining_assignments > 0:
#             assigned_gpus = assigned_gpus + [-1] * remaining_assignments
# 
#         assigned_gpus = assigned_gpus + [0]
# 
#         for i in range(len(layers)):
#             layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)
# 
#     model.gpus = gpus
# 
# 
# def benchmark(model, input_ids, check=False):
#     input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
#     torch.cuda.synchronize()
# 
#     cache = {'past': None}
# 
#     def clear_past(i):
# 
#         def tmp(layer, inp, out):
#             if cache['past']:
#                 cache['past'][i] = None
# 
#         return tmp
# 
#     for i, layer in enumerate(model.model.layers):
#         layer.register_forward_hook(clear_past(i))
# 
#     print('Benchmarking ...')
# 
#     if check:
#         loss = nn.CrossEntropyLoss()
#         tot = 0.
# 
#     def sync():
#         if hasattr(model, 'gpus'):
#             for gpu in model.gpus:
#                 torch.cuda.synchronize(gpu)
#         else:
#             torch.cuda.synchronize()
# 
#     max_memory = 0
#     with torch.no_grad():
#         attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
#         times = []
#         for i in range(input_ids.numel()):
#             tick = time.time()
#             out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
#             sync()
#             times.append(time.time() - tick)
#             print(i, times[-1])
#             if hasattr(model, 'gpus'):
#                 mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
#             else:
#                 mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
#             max_memory = max(max_memory, mem_allocated)
#             if check and i != input_ids.numel() - 1:
#                 tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
#             cache['past'] = list(out.past_key_values)
#             del out
#         sync()
#         print('Median:', np.median(times))
#         if check:
#             print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
#             print('max memory(MiB):', max_memory)


def quantize(model_id: str, bits: int, groupsize: int, output_dir: str, trust_remote_code: bool):
    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="balanced_low_0", trust_remote_code=trust_remote_code)
    print("LOADED model")
    model.seqlen = 2048

    dataset = "wikitext2"
    nsamples = 128
    seed = None


    dataloader, testloader = get_loaders(dataset, nsamples=nsamples, seed=seed, model_id=model_id, seqlen=model.seqlen)

    tick = time.time()
    quantizers = sequential(model, dataloader, DEV, nsamples, bits, groupsize)
    print(time.time() - tick)

    # if args.benchmark:
    #     gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    #     if len(gpus) > 1:
    #         llama_multigpu(model, gpus, gpu_dist)
    #     else:
    #         model = model.to(DEV)
    #     if args.benchmark:
    #         input_ids = next(iter(dataloader))[0][:, :args.benchmark]
    #         benchmark(model, input_ids, check=args.check)

    # if args.eval:
    #     datasets = ['wikitext2', 'ptb', 'c4']
    #     if args.new_eval:
    #         datasets = ['wikitext2', 'ptb-new', 'c4-new']
    #     for dataset in datasets:
    #         dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
    #         print(dataset)
    #         llama_eval(model, testloader, DEV)
    # 
    # if args.test_generation:
    #     gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    #     if len(gpus) > 1:
    #         llama_multigpu(model, gpus, gpu_dist)
    #     else:
    #         model = model.to(DEV)

    #     from transformers import LlamaTokenizer, TextStreamer
    #     tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    #     input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
    #     streamer = TextStreamer(tokenizer)
    #     with torch.no_grad():
    #         generated_ids = model.generate(input_ids, streamer=streamer)
    #     


    # if args.quant_directory is not None:
    #     export_quant_table(quantizers, args.quant_directory)

    # if not args.observe and args.save:
    #     llama_pack(model, quantizers, args.bits, args.groupsize)
    #     torch.save(model.state_dict(), args.save)

    # if not args.observe and args.save_safetensors:
    pack(model, quantizers, bits, groupsize)
    from safetensors.torch import save_file
    from transformers.modeling_utils import shard_checkpoint
    state_dict = model.state_dict()
    state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
    state_dict["gptq_bits"] = torch.LongTensor([bits])
    state_dict["gptq_groupsize"] = torch.LongTensor([groupsize])

    max_shard_size = "10GB"
    shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name="model.safetensors")
    os.makedirs(output_dir, exist_ok=True)
    for shard_file, shard in shards.items():
        save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt", "quantized": "gptq", "origin": "text-generation-inference"})
    if index is None:
        path_to_weights = os.path.join(output_dir, "model.safetensors")
        logger.info(f"Model weights saved in {path_to_weights}")
    else:
        save_index_file = "model.safetensors.index.json"
        save_index_file = os.path.join(output_dir, save_index_file)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(output_dir)
    logger.info("Saved config")
    logger.info("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved tokenizer")
