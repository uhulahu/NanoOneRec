import argparse
import collections
import json
import os
import random
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import numpy as np
from utils import * 
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object

def load_data(args):
    if args.root:
        print("args.root: ", args.root)
    # 2026-09-03 目录重组：item.json 在 raw/ 下；保留旧扁平布局回退（打印实际使用路径）
    for sub in ('raw', ''):
        item2feature_path = os.path.join(args.root, sub, f'{args.dataset}.item.json')
        if os.path.isfile(item2feature_path):
            print(f"load item.json from: {item2feature_path}")
            return load_json(item2feature_path)
    raise FileNotFoundError(f"item.json not found under {args.root} (expected raw/ or flat layout)")

def generate_text(item2feature, features):
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])  # 先进行一个格式处理，长度超过2000的字段直接舍弃（置为空字符串）
                cleaned = meta_value.strip()
                if cleaned != "":
                    text.append(cleaned)

        if len(text) == 0:
            text = ["unknown item"]
        
        try:
            item_id = int(item)
        except:
            item_id = item
            
        item_text_list.append((item_id, " ".join(text)))

    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)
    item2feature = load_data(args)
    item_text_list = generate_text(item2feature, ['title', 'description'])  # 只使用两个特征，title 和 description
    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, accelerator, word_drop_ratio=-1, batch_size=64, pooling='mean'):
    """多进程分块 → 批量前向 → 池化 → 汇聚排序落盘

    pooling: 'mean'（默认，掩码均值池化，BERT 系惯例）或 'last'（last token pooling + L2 归一化，Qwen3-Embedding 官方做法，适合 decoder-only 模型）。
    使用 'last' 时保存文件名带 -last 后缀，避免覆盖 'mean' 的结果。
    """
    if pooling not in ('mean', 'last'):
        raise ValueError(f"pooling must be 'mean' or 'last', got {pooling!r}")
    all_ids, all_texts = zip(*item_text_list)  # 
    
    total_items = len(all_texts)

    # === 数据分片 === 按 item 索引连续分块，每个进程各拿一块
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index  # 获取当前进程索引

    # 获取当前进程所负责的块
    chunk_size = int(np.ceil(total_items / num_processes))  # 每个块的大小
    start_idx = process_index * chunk_size  # int, 获取当前进程所负责的块的起始下标
    end_idx = min(start_idx + chunk_size, total_items)  # int, 获取当前进程所负责的块的终止下标（左闭右开）
    
    local_ids = all_ids[start_idx:end_idx]  # tuple of int, len = chunk_size
    local_texts = all_texts[start_idx:end_idx]  # tuple of str, len = chunk_size

    if accelerator.is_main_process:  # 只在主进程打印
        print(f"Total items: {total_items}")
        print(f"Start generating embeddings ({pooling} pooling) with {num_processes} processes...")

    # === 进程内批量循环 ===
    local_results = []  # 收集embedding

    # 只在主进程打印进度条
    pbar = tqdm(total=len(local_texts), desc=f"Proc {process_index}", disable=not accelerator.is_local_main_process)

    # last token pooling 需配合 left padding；mean pooling 用法不变（right）
    tokenizer.padding_side = "left" if pooling == 'last' else "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen 原始模型常无 pad_token，用 eos 兜底

    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            batch_texts = list(local_texts[i : i + batch_size])
            batch_ids = local_ids[i : i + batch_size]

            # Word Drop Logic (Batch Level)
            if word_drop_ratio > 0:
                processed_batch = []
                for text in batch_texts:
                    sent = text.split(' ')
                    new_sent = [wd for wd in sent if random.random() > word_drop_ratio]
                    processed_batch.append(' '.join(new_sent))
                batch_texts = processed_batch

            # Tokenization
            encoded_sentences = tokenizer(
                batch_texts, 
                max_length=args.max_sent_len,  # 最大输入长度
                truncation=True,   # 按 max_sent_len 截断
                return_tensors='pt', 
                padding=True
            ).to(accelerator.device)

            input_ids = encoded_sentences.input_ids  # [batch, seq]
            attention_mask = encoded_sentences.attention_mask  # [batch, seq]

            # Model Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if pooling == 'mean':
                # Mean Pooling (Masked)
                # outputs.last_hidden_state: [batch, seq, dim]
                last_hidden = outputs.last_hidden_state

                # [batch, seq] -> [batch, seq, 1] -> [batch, seq, dim]
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)  # [batch, dim]
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # [batch, dim]

                # 相当于在seq维度上进行 mean pooling，作为该句子的 embedding
                sentence_output = sum_embeddings / sum_mask  # [batch, dim]
            elif pooling == 'last':
                # Last Token Pooling + L2 Normalize（Qwen3-Embedding 官方做法）
                # decoder-only 模型只有最后一个 token 能看到完整序列
                sentence_output = last_token_pool(outputs.last_hidden_state, attention_mask)  # [batch, dim]
                sentence_output = F.normalize(sentence_output, p=2, dim=1)

            # return to CPU Numpy
            sentence_output = sentence_output.cpu().numpy()

            for idx, emb in zip(batch_ids, sentence_output):
                local_results.append((idx, emb))

            pbar.update(len(batch_texts))
    
    pbar.close()

    # === 汇总到主进程排序落盘 ===
    accelerator.wait_for_everyone()

    all_results_flat = gather_object(local_results)

    if accelerator.is_main_process:
        print("Gathering finished. Sorting and saving...")
        
        all_results_flat.sort(key=lambda x: x[0])  # 按 id 排序
        
        final_embeddings = np.stack([x[1] for x in all_results_flat], axis=0)
        
        print('Final Embeddings shape: ', final_embeddings.shape)
        
        # 'last' pooling 加 -last 后缀，避免覆盖 'mean' 的结果
        suffix = "-td-last" if pooling == 'last' else "-td"
        # 2026-09-03 目录重组：向量统一输出到 emb/ 子目录
        emb_dir = os.path.join(args.root, 'emb')
        os.makedirs(emb_dir, exist_ok=True)
        file_path = os.path.join(emb_dir, f"{args.dataset}.emb-{args.plm_name}{suffix}.npy")
        np.save(file_path, final_embeddings)
        print(f"Saved to {file_path}")

def load_qwen_model(model_path, use_flash_attention=False):
    print("Loading Qwen Model:", model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # AutoModel 返回基座模型，前向输出里带 last_hidden_state
    # AutoModelForCausalLM 前向输出会是 logits
    load_kwargs = dict(
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if use_flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModel.from_pretrained(model_path, **load_kwargs)

    return tokenizer, model

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Qwen3-Embedding 官方推荐的 last token pooling。

    取每个序列"最后一个有效 token"的 hidden state 作为句向量。
    Qwen 是 decoder-only（causal attention）架构，只有最后一个 token 能看到完整序列，因此它的向量包含整句信息（mean pooling 会稀释信息）。
    兼容 left / right 两种 padding：
    - left padding：每个序列最末位恒为真实 token，直接取 last_hidden_states[:, -1]
    - right padding：按 attention_mask 统计每个序列的真实长度，索引出末位 token
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Beauty / Sports / Toys')
    parser.add_argument('--root', type=str, default="")
    # parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--plm_name', type=str, default='qwen')
    parser.add_argument('--plm_checkpoint', type=str, default='xxx', help='Qwen model path')
    parser.add_argument('--pooling', type=str, default='last', help='mean or last')
    parser.add_argument('--max_sent_len', type=int, default=2048, help='输入长度截断')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    accelerator = Accelerator() # 使用accelerator多卡并行推理
    
    if accelerator.is_main_process:
        print(f"Running with {accelerator.num_processes} processes.")

    item_text_list = preprocess_text(args)

    # Pre-trained Language Mode
    plm_tokenizer, plm_model = load_qwen_model(args.plm_checkpoint)  
    
    plm_model = plm_model.to(accelerator.device)
    plm_model.eval()

    generate_item_embedding(
        args, 
        item_text_list, 
        plm_tokenizer, 
        plm_model, 
        accelerator, 
        word_drop_ratio=args.word_drop_ratio,
        pooling=args.pooling
    )
