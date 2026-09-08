# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import apply_chat_template, is_conversational, maybe_apply_chat_template
# from trl import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl import SyncRefModelCallback
from trl import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import random

from transformers import (
        is_wandb_available, 
        AutoTokenizer, 
        AutoModelForCausalLM,
        TemperatureLogitsWarper, 
        LogitsProcessorList,
        Trainer
    )

from LogitProcessor import ConstrainedLogitsProcessor
from transformers.generation import LogitsProcessor
import math
import re

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_vllm_available():
    # from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class MemTrackerCallback(TrainerCallback):
    """显存打点（OOM 调查 2026-09-02）：每 log_every 步打印本步峰值与池占用，随后 empty_cache 让池回落。

    判读：empty_cache 后 allocated 仍随步数涨 = 真泄漏（有对象持有 GPU 张量）；
    仅 reserved 涨而 allocated 回落 = 分配器池/碎片（无泄漏，但每步峰值需要降）。
    """

    def __init__(self, log_every: int = 25, empty_cache: bool = True):
        self.log_every = log_every
        self.empty_cache = empty_cache

    def on_step_begin(self, args, state, control, **kwargs):
        # 每 log_every 步清零峰值统计 → 记录的是"该步本身"的峰值（含 rollout+ref+前向+backward）
        if state.global_step % self.log_every == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0:
            return
        if not torch.cuda.is_available():
            return
        gb = 2**30
        rank = os.environ.get("RANK", "0")
        a = torch.cuda.memory_allocated() / gb
        r = torch.cuda.memory_reserved() / gb
        p = torch.cuda.max_memory_allocated() / gb
        print(f"[mem][rank {rank}] step={state.global_step} allocated={a:.2f}GiB "
              f"reserved={r:.2f}GiB peak_this_step={p:.2f}GiB", flush=True)
        if self.empty_cache:
            torch.cuda.empty_cache()
            a2 = torch.cuda.memory_allocated() / gb
            r2 = torch.cuda.memory_reserved() / gb
            print(f"[mem][rank {rank}] step={state.global_step} after empty_cache: "
                  f"allocated={a2:.2f}GiB reserved={r2:.2f}GiB", flush=True)


class ArchiveCheckpointCallback(TrainerCallback):
    """checkpoint 自动归档（防 save_total_limit 轮换删掉对照锚点）。

    全量 run（3750 步）里 save_total_limit=3 会在第 5 次保存（step ~1875）自动删除早期
    ckpt-750——它是与 fd/baseline 同歩数直接对照（RL_IDEAS.md A/B 协议）的锚点。
    本回调在指定 step 的 ckpt 刚保存完（on_save）时立即把整个目录移出 run 目录到
    <archive_dir>/<run名>/checkpoint-N（同盘 rename，瞬时完成），此后轮换逻辑不再可见它。
    移出不改 trainer_state/轮换计数，也不影响从其他 ckpt resume；移出的目录可直接作为
    evaluate_rl.sh 的模型路径使用（内含 config.json/tokenizer/optimizer.pt，也可 resume）。
    """

    def __init__(self, steps=(750,), archive_dir="ckpt_archive"):
        self.steps = set(int(s) for s in steps)
        self.archive_dir = archive_dir

    def on_save(self, args, state, control, **kwargs):
        if state.global_step not in self.steps:
            return
        # 只在主进程执行（多进程共享文件系统，防止竞态重复 move）
        if not (getattr(state, "is_world_process_zero", False) or getattr(state, "is_local_process_zero", False)):
            return
        src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        run_name = os.path.basename(os.path.normpath(args.output_dir))
        dst = os.path.join(self.archive_dir, run_name, f"checkpoint-{state.global_step}")
        if os.path.isdir(src) and not os.path.isdir(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f"[archive] checkpoint-{state.global_step} -> {dst}", flush=True)


def build_sid_hash_tries(info_file: str, base_model: str):
    """构建 per-task 双 trie（2026-09-03，动机与设计见 docs/RL_IDEAS.md）。

    - hash_dict_full：全长 trie（NTP 任务用）。每个 item 的完整 sid 路径（碰撞 item 含 <d_x>），
      EOS 只挂在完整路径末端——碰撞前缀在 3 级后只允许续 <d_x>。
    - hash_dict_prefix：前缀 trie（对齐任务 title/desc→sid 用）。每条 sid 先截断到前 3 级
      （与 RLTitle2SidDataset 的 target 同口径）再建树 → 任意 3 级前缀后即可停（\n→EOS），
      <d_*> 不可达。全长 trie 下对齐样本的 3 级答案永远不在解码支持集、被迫生成的 <d_x> 被
      first-diff 误判 -1（支持集铁律 violation），前缀 trie 让答案回到支持集。
    两棵树在 3 级之前（含 unique 前缀的 3 级节点）结构完全相同，仅碰撞前缀的 3 级节点分叉。

    返回 (hash_dict_full, hash_dict_prefix, max_sid)。max_sid = 全场 sid 级数上限。
    """
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Parse new format: semantic_id \t item_title \t item_id
        full_sids = [line.split('\t')[0].strip() for line in info]
        # 截断到前 3 级（前缀 trie 用；与对齐任务 target 同口径，extra <d_x> 是桶内身份编号、无语义）
        prefix3_sids = ["".join(re.findall(r"<[abcd]_\d+>", sid)[:3]) for sid in full_sids]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # 现场统计 SID 级数上限（RQ 量化级数，约束解码保证生成不超过此级数）
    max_sid = max((len(re.findall(r"<[abcd]_\d+>", sid)) for sid in full_sids), default=4)

    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3

    def _tokenize_sid_lines(sid_list):
        """每条 sid 按 '### Response:\\n{...}\\n' 模板 tokenize（行尾带 \\n；首级 key = 模板尾，与 decode 行尾对齐）。"""
        lines = [f"### Response:\n{_}\n" for _ in sid_list]
        if base_model.lower().find("llama") > -1:
            return [tokenizer(_).input_ids[1:] for _ in lines]  # llama: 去掉 BOS
        return [tokenizer(_).input_ids for _ in lines]

    def _build_hash_dict(ids_list):
        """前缀树：key = 已生成前缀（首级 key = 模板尾 ID[:prefix_index]，其后 = sid tokens），
        value = 该节点允许的下一 token 列表（含行进到路径末端时追加的 \\n / EOS）。"""
        d = {}
        for ID in ids_list:
            ID.append(tokenizer.eos_token_id)
            for i in range(prefix_index, len(ID)):
                if i == prefix_index:
                    hash_number = '-'.join(str(_) for _ in ID[:i])
                else:
                    hash_number = '-'.join(str(_) for _ in ID[prefix_index:i])
                d.setdefault(hash_number, set()).add(ID[i])
        return {k: list(v) for k, v in d.items()}

    hash_dict_full = _build_hash_dict(_tokenize_sid_lines(full_sids))
    hash_dict_prefix = _build_hash_dict(_tokenize_sid_lines(prefix3_sids))
    return hash_dict_full, hash_dict_prefix, max_sid


class ReReTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method adapted to recommendation. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        base_model: str,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,

        #* sample
        add_gt: bool = False,
        dynamic_sampling: bool = False,
        beam_search: bool = False,
        length_penalty: float = 0.0,
        #* eval
        test_during_training: bool = True,
        test_beam: int = 20,

        #*loss
        dapo: bool = False,
        gspo: bool = False,

        #* token 级 advantage 归一化（想法 (b)(c)，见 docs/RL_IDEAS.md）
        token_norm: str = "group",        # "group"=组内列 z（原实现）| "column"=跨组列 z
        all_wrong_penalty: float = 0.0,   # >0：全错列附加惩罚 λ（默认 0 = 关）

        #* others
        info_file: str = None,
        # per-task 双 trie（2026-09-03）：对齐任务（title/desc→sid）prompt 集合 → 解码用前缀 trie（3 级即停、
        # <d_x> 不可达）；集合外（NTP/eval）用全长 trie。None/缺省 = 全部全长（旧行为，rl_gpr 等未接入处不变）
        trie_prefix_prompts: Optional[set] = None, 
        # logits_processor: Optional[LogitsProcessor] = None,
        prompt2history: dict[str, str] = None,
        history2target: dict[str, str] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # === 兼容补丁：trl 1.12 的 GRPOConfig 已移除旧版字段，补默认值防止 AttributeError（详见 MONITORING_LOG.md 2026-09-02）===
        for _attr, _default in [
            ("max_prompt_length", None),
            ("vllm_device", "auto"),
            ("vllm_dtype", "auto"),
            ("vllm_max_model_len", None),
        ]:
            if not hasattr(args, _attr):
                setattr(args, _attr, _default)

        # Models
        # Trained model
        self.base_model = base_model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(self.base_model, padding_side="left")
            processing_class.pad_token = processing_class.eos_token


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        print(f"max_completion_length: {self.max_completion_length}")
        self.num_generations = args.num_generations  # = G in the GRPO paper 
        self.use_vllm = args.use_vllm

        self.beta = args.beta
        

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)   
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.prompt2history = prompt2history
        self.history2target = history2target
        self.add_gt = add_gt
        self.beam_search = beam_search
        self.info_file = info_file
        self._trie_prefix_prompts = trie_prefix_prompts or set()
        self._row_trie_kinds = None  # 每次 rollout 前由 _prepare_inputs 设置：本地行 → 0=全长/1=前缀 trie
        self.temperature = args.temperature
        self.length_penalty = length_penalty
        self.test_during_training = test_during_training
        self.test_beam = test_beam
        self.dynamic_sampling = dynamic_sampling
        self.dapo = dapo
        self.gspo = gspo
        if token_norm not in ("group", "column"):
            raise ValueError(f"token_norm 只能取 'group'/'column'，实际为 {token_norm!r}")
        self.token_norm = token_norm
        self.all_wrong_penalty = float(all_wrong_penalty)
        # self.logits_processor = logits_processor

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            if self.beam_search:
                 #* temperature 默认为 1.0
                print(f"self.temperature: {self.temperature}")
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    length_penalty=self.length_penalty,
                    num_beams=self.num_generations,
                    num_return_sequences=self.num_generations,
                    pad_token_id=processing_class.pad_token_id,
                    eos_token_id=processing_class.eos_token_id,
                    top_k=None,
                    top_p=None,
                    temperature=self.temperature,
                    # 确定性束搜索（论文 3.4.1 口径）：do_sample=False 保证同 prompt 每次 rollout 相同的 16 条 beam，
                    # 且与 test_generation_config（574 行）一致。原为 True（采样束），随机性使训练信号不稳定。
                    do_sample=False,
                )
            else:
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    length_penalty=self.length_penalty,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=processing_class.pad_token_id,
                    eos_token_id=processing_class.eos_token_id,
                )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            # print("Sync Begin")
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # === 构建 trie 约束（per-task 双 trie，2026-09-03，动机/设计见 build_sid_hash_tries 与 docs/RL_IDEAS.md）===
        # 全长 trie = NTP（历史→全长 sid）与 eval；前缀 trie = 对齐任务（title/desc→sid，target 只到 3 级）。
        # 每次 rollout 由 prefix_allowed_tokens_fn 按行（_row_trie_kinds）选择。
        self.hash_dict_full, self.hash_dict_prefix, self.max_sid = build_sid_hash_tries(self.info_file, self.base_model)
        self.hash_dict = self.hash_dict_full  # 默认引用 = 全长 trie（eval/beam/未接 kinds 的场景回退）

        self.test_generation_config = GenerationConfig(max_new_tokens=self.max_completion_length,
                                                            length_penalty=self.length_penalty,
                                                            num_beams=self.test_beam,
                                                            num_return_sequences=self.test_beam,
                                                            do_sample=False,
                                                            top_k=None,
                                                            top_p=None,
                                                            pad_token_id=self.processing_class.pad_token_id,
                                                            eos_token_id=self.processing_class.eos_token_id,)

    def get_hash(self, x):
            x = [str(_) for _ in x]
            return '-'.join(x)

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        # per-task 双 trie：按 rollout 行（batch_id）选约束表。
        # kind 0 = 全长 trie（NTP/eval/默认）；1 = 前缀 trie（对齐任务，3 级即停、<d_x> 不可达）。
        # 采样路径 batch_id = 本地行号；beam 路径（dedup 后每 prompt G 条 beam 连续）batch_id = prompt 序号，
        # _row_trie_kinds 均等长对齐（见 _prepare_inputs）。越界（eval test-beam 20 束 ≠ 16 行布局）回退全长 trie。
        kind = 0
        if self._row_trie_kinds is not None and batch_id < len(self._row_trie_kinds):
            kind = self._row_trie_kinds[batch_id]
        table = self.hash_dict_prefix if kind == 1 else self.hash_dict
        hash_number = self.get_hash(input_ids)
        if hash_number in table:
            return table[hash_number]
        return []
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # def _get_train_sampler(self,  *args, **kwargs) -> Sampler:
    #     # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
    #     # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
    #     # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
    #     # preventing discrepancies in group formation.
    #     sampler = super()._get_train_sampler(*args, **kwargs)
    #     return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)
    
    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        if train_dataset is None:
            train_dataset = self.train_dataset
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                unwrapped_model.unmerge_adapter()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        # per-task 双 trie：本地 rollout 的任务类型 kinds（1=对齐任务→前缀 trie，0=全长 trie），
        # 供 prefix_allowed_tokens_fn 按 batch_id 查表。两种生成路径的下标语义不同：
        # - 采样路径：每输入行 → 一条输出序列，行序不变 → kinds 按行存（长度 = len(prompts)）。
        # - beam 路径（实际 RL run 的生成方式，rl.sh --beam_search True）：rollout 先 dedup 每 prompt
        #   取 1 行（i%G==0），generate 内部按 num_beams=G 展开、每 prompt 的 G 条 beam 连续 → processor
        #   的 batch_id = dedup 后的 prompt 序号（0..P-1）→ kinds 按 dedup prompts（prompts[::G]）存。
        #   eval 的 test-beam（num_beams=test_beam=20）行数与 16 不对齐，但 eval 全是 NTP（kind 0），
        #   查错/越界只可能得到 0（全长 trie），无影响。
        if self.beam_search:
            self._row_trie_kinds = [1 if p in self._trie_prefix_prompts else 0
                                    for p in prompts[::self.num_generations]]
        else:
            self._row_trie_kinds = [1 if p in self._trie_prefix_prompts else 0 for p in prompts]

        if self.add_gt or self.test_during_training or self.dynamic_sampling:
            histories = [self.prompt2history[x["prompt"]] for x in inputs]
            targets = [self.history2target[x] for x in histories]
            # print(f"targets: {targets}")
            num_categories = len(set(targets)) 
        # target_ids = self.processing_class(targets, return_tensors="pt", padding=True, padding_side="left")["input_ids"]
        # target_ids = target_ids.to(device)
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        ccc = ConstrainedLogitsProcessor(
                # guidance_scale=1.0,
                # cf_logits=None,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                # cf_dict=sasrec_dict,
                # unconditional_ids=None,
                num_beams=self.num_generations if self.beam_search else 1,
                base_model=self.base_model,
                eos_token_id=self.processing_class.eos_token_id
            )
        self.logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=self.temperature), ccc])
        self.test_lp_list = LogitsProcessorList([ccc])

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                topk = [3, 5, 10, 20]
                ndcg = [0 , 0, 0, 0]
                hr = [0, 0, 0, 0]

                # 测试生成只在 eval 步执行（model.eval()）：训练步跳过，避免每步一次 beam20 测试的开销；
                # 且 eval 步的 batch 来自 eval_dataset → 指标是独立 eval 集的 HR/NDCG（无训练集自评偏差）
                if self.test_during_training and not self.model.training:
                    dedup_prompt = []
                    dedup_mask = []
                    dedup_target = []

                    for i in range(len(prompt_ids)):
                        if i % self.num_generations == 0:
                            dedup_prompt.append(prompt_ids[i])
                            dedup_mask.append(prompt_mask[i])
                            dedup_target.append(targets[i])
                    
                    dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                    dedup_prompt_mask = torch.stack(dedup_mask).to(device)
                    # print(f"dedup_prompt_ids: {dedup_prompt_ids.shape}")
                
                    # print(f"test_beam: {self.test_beam}")
                    with torch.no_grad():
                        test_completion_ids = unwrapped_model.generate(
                            dedup_prompt_ids, attention_mask=dedup_prompt_mask, generation_config=self.test_generation_config,
                            logits_processor=self.test_lp_list,
                            use_model_defaults=False,  # 防止 model.config 默认值(do_sample=True,temperature=0.6)覆盖显式配置
                        )
                    
                    # print(f"test_completion_ids: {test_completion_ids.shape}")
                    if self.base_model.lower().find("llama")>-1:
                        test_completions = self.processing_class.batch_decode(test_completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    else:
                        test_completions = self.processing_class.batch_decode(test_completion_ids, skip_special_tokens=True)
                    test_completions = [_.split("Response:\n")[-1] for _ in test_completions]
                    test_comp_lis = [test_completions[i:i+self.test_beam] for i in range(0, len(test_completions), self.test_beam)]
                    for i, comp_lis in enumerate(test_comp_lis):
                        target = dedup_target[i]
                        for j in range(len(comp_lis)):
                            if comp_lis[j].strip("\n\"") == target.strip("\n\""):
                                for index, k in enumerate(topk):
                                    if j < k:
                                        hr[index] += 1
                                        ndcg[index] += 1 / math.log2(j+2) 
                                break
                    hr = [elm/len(dedup_target) for elm in hr]
                    ndcg = [elm/len(dedup_target) for elm in ndcg]

                if self.beam_search:
                    dedup_prompt = []
                    dedup_mask = []
                    for i in range(len(prompt_ids)):
                        if i % self.num_generations == 0:
                            dedup_prompt.append(prompt_ids[i])
                            dedup_mask.append(prompt_mask[i])
                    dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                    dedup_prompt_mask = torch.stack(dedup_mask).to(device)
                    # print(f"dedup_prompt_ids: {dedup_prompt_ids.shape}")
                    prompt_completion_ids = unwrapped_model.generate(
                        dedup_prompt_ids, attention_mask=dedup_prompt_mask, generation_config=self.generation_config,
                        logits_processor=self.logits_processor,
                        use_model_defaults=False,  # 防止 model.config 默认值(do_sample=True,temperature=0.6)覆盖显式配置
                    )
                    # print(f"prompt_ids: {prompt_ids.shape}")
                    # print(f"prompt_completion_ids: {prompt_completion_ids.shape}")
                else:
                    if self.dynamic_sampling:
                        lis1 = []
                        lis2 = []
                        extended_targets = []
                        for i in range(0, len(prompt_ids), self.num_generations):
                            lis1.extend([prompt_ids[i]]*int(1.5*self.num_generations))
                            lis2.extend([prompt_mask[i]]*int(1.5*self.num_generations))
                            extended_targets.extend([targets[i]]*int(1.5*self.num_generations))
                        extended_prompt_ids = torch.stack(lis1).to(device)
                        extended_prompt_mask = torch.stack(lis2).to(device)
                        # dynamic sampling 把每 prompt 的行扩到 1.5×G（prompt 主序）→ 行 kinds 同步扩展
                        #（每 block 取首行 kind 重复 1.5×G 次，与 extended 行序一致）
                        self._row_trie_kinds = [
                            k for k in self._row_trie_kinds[::self.num_generations]
                            for _ in range(int(1.5 * self.num_generations))]
                        # print(f"extended_prompt_ids: {extended_prompt_ids.shape}")
                        # print(f"extended_prompt_mask: {extended_prompt_mask.shape}")
                        prompt_completion_ids = unwrapped_model.generate(
                            extended_prompt_ids, attention_mask=extended_prompt_mask, generation_config=self.generation_config,
                            logits_processor=self.logits_processor,
                            use_model_defaults=False,  # 防止 model.config 默认值(do_sample=True,temperature=0.6)覆盖显式配置
                        )
                        prompt_length = prompt_ids.size(1)
                        extended_completion_ids = prompt_completion_ids[:, prompt_length:]
                        if self.base_model.lower().find("llama")>-1:
                            extended_completions_text = self.processing_class.batch_decode(extended_completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        else:
                            extended_completions_text = self.processing_class.batch_decode(extended_completion_ids, skip_special_tokens=True)
                        # print(f"extended_completions_text: {extended_completions_text}")

                        def select_completion(completions, target):
                            from collections import Counter
                            selected = []
                            completion_times = Counter(completions)
                            completion_times = dict(sorted(completion_times.items(), key=lambda x: x[1], reverse=True))
                            if target in completions:
                                selected.extend([target]*min(completion_times[target], self.num_generations))
                            if len(selected) == self.num_generations:
                                return selected
                            for item in completion_times:
                                if item != target and completion_times[item] > 0:
                                    selected.append(item)
                                    completion_times[item] -= 1
                                    if len(selected) == self.num_generations:
                                        return selected
                            while len(selected) < self.num_generations:
                                for item in completion_times:
                                    if item != target and completion_times[item] > 0:
                                        selected.append(item)
                                        completion_times[item] -= 1
                                        if len(selected) == self.num_generations:
                                            return selected
                        selected_completion = []
                        for i in range(0, len(extended_completions_text), int(self.num_generations*1.5)):
                            selected_completion.extend(select_completion(extended_completions_text[i:i+int(self.num_generations*1.5)], extended_targets[i]))
                        # print(f"selected_completion: {len(selected_completion)}")
                        selected_completion_ids = self.processing_class(selected_completion, return_tensors="pt", padding=True, padding_side="right", \
                            add_special_tokens=True)["input_ids"].to(device)
                        # print(f"selected_completion_ids: {selected_completion_ids.shape}")
                        prompt_completion_ids = torch.cat([prompt_ids, selected_completion_ids], dim=1)
                        # print(f"dynSam_prompt_completion_ids: {prompt_completion_ids.shape}")
                            
                    else:
                        prompt_completion_ids = unwrapped_model.generate(
                            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config,
                            logits_processor=self.logits_processor,
                            use_model_defaults=False,  # 防止 model.config 默认值(do_sample=True,temperature=0.6)覆盖显式配置
                        )

            if self.add_gt:
                repeat = len(prompts) // num_categories
                new_prompt_completions = []
                flag = False
                # rep_ind = [random.randint(i, i+repeat-1) for i in range(0, len(prompts), repeat)]
                for i in range(len(prompts)):
                    if (i+1)%repeat == 0:
                        target_ids = self.processing_class(targets[i], return_tensors="pt", padding=True, padding_side="left", \
                            add_special_tokens=True)["input_ids"].squeeze()
                        # print(f"target_ids: {target_ids.shape}")
                        # print(f"prompt_ids: {prompt_ids[idx].shape}")
                        target_ids = target_ids.to(device)
                        added_ids = torch.cat([prompt_ids[i], target_ids], dim=0)
                        # print(f"added_ids: {added_ids.shape}")
                        new_prompt_completions.append(added_ids)
                    else:
                        new_prompt_completions.append(prompt_completion_ids[i])
                prompt_completion_ids = pad(new_prompt_completions, padding_value=self.processing_class.pad_token_id)
                
                    
            prompt_length = prompt_ids.size(1)
            # 生成/rollout 只当值用：在此切断 autograd 图。train() 下权重 requires_grad，不 detach 的话
            # beam-search 整条前向图的 saved tensors 会活到 loss.backward()，与 ref/teacher-forcing 前向
            # 叠加把每步 live 显存抬到 ~3× 前向（OOM 调查 2026-09-02）
            prompt_completion_ids = prompt_completion_ids.detach()
            prompt_ids = prompt_completion_ids[:, :prompt_length]

            completion_ids = prompt_completion_ids[:, prompt_length:]


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(completions_text)
        
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        if self.base_model.lower().find("llama")>-1:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(completions_text)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        div_lis = [len(set(completions_text[i:i+self.num_generations]))/self.num_generations for i in range(0, len(completions_text), self.num_generations)]
        # cate_diversity = len(set(completions_text))/len(completions_text)
        cate_diversity = sum(div_lis)/len(div_lis)
        completion_ids_cpu = completion_ids.cpu().numpy()
        total_ids = set()
        num_tokens = 0
        for ids in completion_ids_cpu:
            ids = ids[ids != self.processing_class.pad_token_id]
            total_ids.update(set(ids))
            num_tokens += len(ids)        
        num_unique_tokens = len(total_ids)
        token_diversity = num_unique_tokens / num_tokens if num_tokens > 0 else 0.0

        # === 计算奖励 ===
        # 创建一个空的奖励 tensor，[B, num_reward]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        # token 级奖励（标记 token_level=True）：返回 list of {"scores":[...], "mask":[...]}，单独收集，不走标量 rewards_per_func 列
        token_scores_raw = None
        token_masks_raw = None
        token_reward_idx = None  # token 级函数在 reward_funcs 中的位置（用于取 reward_weights 对应权重）
        # 遍历每个奖励函数
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                # 处理 token-level 奖励
                if getattr(reward_func, "token_level", False):
                    if token_scores_raw is not None:
                        raise ValueError("目前仅支持 0/1 个 token 级奖励函数（token_level=True）")
                    token_scores_raw = [r["scores"] for r in output_reward_func]
                    token_masks_raw = [r["mask"] for r in output_reward_func]
                    token_reward_idx = i  # 记录位置：合并处取 reward_weights[token_reward_idx] 作为 token 信号权重
                    continue  # token 级不填标量列（该列保持 0，不参与标量加权求和）
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # === 组内相对优势估计之标量奖励 ===
        # 从各进程收集结果，Gather the reward per function: this part is crucial, because the rewards are normalized per group and the completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        # 对标量奖励加权求和
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)  # [BxG]
        # 计算组内均值方差，组 = 同一个 prompt (query) 的 num_generations 条生成
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  # [B, G]
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # 组内归一化：相当于以组内平均奖励作为 baseline，鼓励那些相对 baseline 优势为正的 output，惩罚为负的
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # print(f"advantages: {advantages}")

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        sliced_rewards = rewards[process_slice]

        # === 优势估计之per-token奖励（first-diff 等）===
        # 5 列结构 = SID 位 0-3 + EOS 停止位（RQ-KMeans 最多 4 级 SID，约束解码下每行自 token 0 起即 SID）；
        # 每组按位置做 masked z-score（只在该位置有监督的成员间归一化，无监督位 advantage=0）；
        # 随后映射到 completion_ids 的 token 空间（SID 位 j → 行内列 j；EOS 位 → 该行 eos token 位置）。
        if token_scores_raw is not None:
            if self.dapo or self.gspo:
                raise ValueError("token 级奖励与 dapo/gspo 变体不兼容（token advantage 无法按序列聚合）")
            token_adv = self._compute_token_advantages(token_scores_raw, token_masks_raw, eos_idx, completion_ids.size(1))
            # 与标量 reward 的 advantage 合并：标量 adv 广播到每个 token + token 级 adv（各自已组内 z-score，尺度一致）。
            token_weight = self.reward_weights[token_reward_idx].to(device) if token_reward_idx is not None else 1.0
            advantages = advantages.unsqueeze(1).expand(-1, completion_ids.size(1)) + token_weight * token_adv

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            if getattr(reward_func, "token_level", False) and token_scores_raw is not None:
                # token 级奖励：记录"有监督位平均分"（+1 前缀 / -1 分歧的平均），便于观察
                flat_s = [s for r in token_scores_raw for s in r]
                flat_m = [m for r in token_masks_raw for m in r]
                masked_vals = [s for s, m in zip(flat_s, flat_m) if m]
                # 用于日志打印
                self._metrics[f"rewards/{reward_func_name}"].append(
                    sum(masked_vals) / len(masked_vals) if masked_vals else 0.0
                )
            else:
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())


        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["categorical_diversity"].append(cate_diversity)
        self._metrics["token_diversity"].append(token_diversity)

        # 与 753 行 test 块同条件：仅 eval 步写入（ndcg/hr 在 eval 步才定义）
        if self.test_during_training and not self.model.training:
            for i in range(len(topk)):
                self._metrics[f"NDCG@{topk[i]}"].append(ndcg[i])
                self._metrics[f"HR@{topk[i]}"].append(hr[i])

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "sliced_rewards": sliced_rewards,
        }
    

    @staticmethod
    def _masked_column_advantages(group_scores, group_masks, mode="group", all_wrong_penalty=0.0, eps=1e-4):
        """masked 逐列标准化（纯函数，供 _compute_token_advantages 与单测共用；想法 (b)(c) 见 docs/RL_IDEAS.md）。

        group_scores: [num_groups, G, C]，监督位 ∈ {+1, -1}，屏蔽位 0
        group_masks:  [num_groups, G, C]，1 = 该位有监督

        mode="group"（默认，即原实现）：每列 mean/std 只在**该组**有监督成员上统计——
          组内同列值全相同（典型：全组在同一位置分歧，列全 -1）时 z=0，该列信号被组基线整体抵消。
        mode="column"（想法 c）：每列 mean/std 在**全 batch 所有组**的有监督成员上统计——
          该组全错而其他组有对的列恢复跨组对比，组内全错不再抵消。

        all_wrong_penalty > 0（想法 b）：z-score 之外，对"该组该列监督成员全为 -1"的 (组,列)
          给每个成员附加 -λ——补回被组基线抹掉的 anti-信号：压低 16 个已生成候选，
          经 softmax 重归一化把质量推向未生成候选（含潜在正确 token）。与 mode="column" 正交可叠加。

        返回 [num_groups, G, C]（屏蔽位恒 0）。
        """
        if mode == "group":
            cnt = group_masks.sum(1).clamp(min=1)                    # [G, C]
            mean = (group_scores * group_masks).sum(1) / cnt
            var = ((group_scores - mean.unsqueeze(1)) ** 2 * group_masks).sum(1) / cnt
            std = var.sqrt()
            adv = (group_scores - mean.unsqueeze(1)) / (std.unsqueeze(1) + eps)
        elif mode == "column":
            cnt = group_masks.sum((0, 1)).clamp(min=1)               # [C]
            mean = (group_scores * group_masks).sum((0, 1)) / cnt
            var = ((group_scores - mean) ** 2 * group_masks).sum((0, 1)) / cnt
            std = var.sqrt()
            adv = (group_scores - mean) / (std + eps)
        else:
            raise ValueError(f"token_norm 只能取 'group'/'column'，实际为 {mode!r}")
        adv = adv * group_masks                                      # 屏蔽位恒 0
        if all_wrong_penalty != 0.0:
            neg_cnt = ((group_scores == -1).float() * group_masks).sum(1)  # [G, C] 监督且为 -1 的条数
            sup_cnt = group_masks.sum(1)                                    # [G, C]（不 clamp：需区分无监督列）
            all_neg = (sup_cnt > 0) & (neg_cnt == sup_cnt)                  # 该列有监督成员全 -1
            adv = adv - all_wrong_penalty * all_neg.unsqueeze(1)            # 广播到组内每个成员
        return adv

    def _compute_token_advantages(self, token_scores_raw, token_masks_raw, eos_idx, completion_len):
        """token 级奖励（如 first-diff）→ token 级组内 advantage。

        输入（每条本地序列一个元素）：
          token_scores_raw[i]: list，长度 = 该序列 SID 数 + 1（末位 = EOS/停止位），元素 ∈ {+1, 0, -1}
          token_masks_raw[i]:  list，与 scores 等长，1 = 该位有监督（+1 前缀 / -1 分歧位），0 = 屏蔽位
          eos_idx:             本地每条 completion 的 EOS token 位置（无 EOS 行 = completion_len）
          completion_len:      completion_ids pad 后的长度

        中间张量列布局：SID 位 0..self.max_sid-1 + EOS 停止位（self.max_sid 列），
        列数 = self.max_sid + 1。约束解码保证生成级数 ≤ self.max_sid，故直接按长度写入无需截断。
        归一化模式见 _masked_column_advantages：group=组内 masked z-score（默认，原实现）；
        column=跨组列标准化（想法 c）；all_wrong_penalty>0=全错列附加惩罚（想法 b）。
        无监督位 advantage 恒为 0（不给信号）。
        返回 [n_local, completion_len]（映射回 completion token 空间：SID 位 j → 行内列 j；EOS 位 → eos token 位置）。
        """
        device = self.accelerator.device
        num_local = len(token_scores_raw)
        eos_col = self.max_sid          # EOS/停止位所在列
        num_cols = self.max_sid + 1
        sid_scores = torch.zeros(num_local, num_cols, device=device)
        sid_masks = torch.zeros(num_local, num_cols, device=device)
        for i in range(num_local):
            raw_scores = token_scores_raw[i]
            raw_masks = token_masks_raw[i]
            num_sid_tokens = len(raw_scores) - 1   # 末位是 EOS/停止位，其余是 SID token
            for j in range(num_sid_tokens):
                sid_scores[i, j] = raw_scores[j]
                sid_masks[i, j] = raw_masks[j]
            sid_scores[i, eos_col] = raw_scores[num_sid_tokens]
            sid_masks[i, eos_col] = raw_masks[num_sid_tokens]
        # 跨进程汇总（组内归一化需同组完整；各进程 shape 相同 [num_local, num_cols]，gather 安全）
        gathered_scores = self.accelerator.gather(sid_scores)
        gathered_masks = self.accelerator.gather(sid_masks)
        num_groups = gathered_scores.shape[0] // self.num_generations
        group_scores = gathered_scores.view(num_groups, self.num_generations, num_cols)
        group_masks = gathered_masks.view(num_groups, self.num_generations, num_cols)
        # masked 逐列标准化 + (b) 全错列惩罚：mode 与惩罚系数由 self.token_norm / self.all_wrong_penalty 控制
        #（group=原实现；column=跨组列标准化；all_wrong_penalty>0=全错列附加惩罚。详见 docs/RL_IDEAS.md）
        token_adv_semantic = self._masked_column_advantages(
            group_scores, group_masks,
            mode=self.token_norm, all_wrong_penalty=self.all_wrong_penalty,
        ).view(-1, num_cols)  # 无监督位 = 0（纯函数内已乘 mask）
        # slice 回本地（与 prompts 行序一致）
        local_slice = slice(
            self.accelerator.process_index * num_local,
            (self.accelerator.process_index + 1) * num_local,
        )
        token_adv_semantic = token_adv_semantic[local_slice]
        # 把 EOS 上的值放回到真实位置
        token_adv = torch.zeros(num_local, completion_len, device=device)
        token_adv[:, :self.max_sid] = token_adv_semantic[:, :self.max_sid]  # SID 位 j → 行内列 j（约束解码下行 0 起即 SID）
        has_eos = eos_idx < completion_len  # 行内有 EOS（自然结束，非截断）
        if has_eos.any():
            token_adv[has_eos, eos_idx[has_eos]] += token_adv_semantic[has_eos, eos_col]
        return token_adv

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")


        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # 已经生成好的完整序列
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # 评估前向，以已经生成的序列作为teacher-forcing，获得 per-token logp（原始分布，无trie约束和mask后的归一化）
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]  # 标量奖励时为 [B*G]（序列级）；token 级奖励（first-diff）时为 [B*G, C]
        # token 级 advantage 已是 2D（每 token 一格）；序列级 1D 需 unsqueeze 广播到 token 维。
        # 注意不改写 advantages：gspo 分支需要原始序列级 1D advantage（s_score·adv 逐样本相乘）。
        per_token_adv = advantages if advantages.dim() == 2 else advantages.unsqueeze(1)

        # ref 与最新策略之间的 kl 散度，K3估计器
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # surrogate；on-policy 下生成和更新之间 θ 不变（不需要修正系数） → ratio 数值恒为 1；不加 clip
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * per_token_adv
        # surrogate - β·kl
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        if self.dapo:
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        elif self.gspo:
            per_token_ratio = per_token_logps - per_token_logps.detach()
            s_score = torch.exp((per_token_ratio*completion_mask).sum(dim=1)/completion_mask.sum(dim=1)) 
            sequence_kl = (per_token_kl * completion_mask).sum(dim=1)/completion_mask.sum(dim=1)
            loss = -(s_score*advantages - self.beta*sequence_kl).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # Log the metrics

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
                
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
