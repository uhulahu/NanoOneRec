"""端到端验证 first_diff_reward + _compute_token_advantages 的列布局消费逻辑。
first_diff_reward 从 rl.py 原样提取函数体；_compute_token_advantages 用桩对象调真实实现源码。
"""
import re
import torch

# ---------- 1. 从 rl.py 原样提取 first_diff_reward（行 296-322） ----------
src = open("/root/autodl-tmp/MiniOneRec-main/rl.py").read().splitlines()
start = next(i for i, l in enumerate(src) if "def first_diff_reward" in l)
end = next(i for i, l in enumerate(src[start:], start) if l.strip() == "return results")
body = "\n".join(l[4:] if l.startswith("    ") else l for l in src[start:end + 1])  # 去 4 空格缩进
history2target = {"h_t4": "<a_1><b_2><c_3><d_4>", "h_t3": "<a_1><b_2><c_3>"}
prompt2history = {"p_t4": "h_t4", "p_t3": "h_t3"}
env = {"re": re, "history2target": history2target, "prompt2history": prompt2history}
exec(body, env)
first_diff_reward = env["first_diff_reward"]

# ---------- 2. 从 minionerec_trainer.py 原样提取 _compute_token_advantages ----------
txt = open("/root/autodl-tmp/MiniOneRec-main/minionerec_trainer.py").read()
m = re.search(r"    def _compute_token_advantages\(.*?\n    def compute_loss\(", txt, re.S)
dedented = "\n".join(l[4:] if l.startswith("    ") else l for l in m.group(0).splitlines()[:-1])


class FakeAccelerator:
    device = torch.device("cpu")
    process_index = 0
    def gather(self, t):
        return t


class FakeSelf:
    max_sid = 4            # 数据集含 4 级 SID
    num_generations = 4
    accelerator = FakeAccelerator()
    def __init__(self):
        env = {"torch": torch, "self": self}
        exec(dedented, env)
        self._fn = env["_compute_token_advantages"]  # 实例属性不绑定 self，调用时显式传

    def compute_token_advantages(self, *args):
        return self._fn(self, *args)


def run_pipeline(rows, completion_len, eos_list):
    """rows: [(prompt, completion_text), ...] 按组排列；eos_list 与该行 SID 后紧跟 EOS 对齐"""
    out = first_diff_reward([r[0] for r in rows], [r[1] for r in rows])
    scores_raw = [o["scores"] for o in out]
    masks_raw = [o["mask"] for o in out]
    eos_idx = torch.tensor(eos_list, dtype=torch.long)
    fake = FakeSelf()
    token_adv = fake.compute_token_advantages(scores_raw, masks_raw, eos_idx, completion_len)
    print(f"{'completion':<28} {'len(s/m)':<8} 语义列填充(分值/监督)                完成 token 空间 advantage")
    for i, row in enumerate(rows):
        sc, mk = scores_raw[i], masks_raw[i]
        cols = []
        for j in range(fake.max_sid + 1):
            if j == fake.max_sid:
                cols.append(f"E:{sc[-1]:.0f}/{mk[-1]}")
            else:
                s = f"{sc[j]:.0f}" if j < len(sc) else "·"
                m = f"{mk[j]}" if j < len(mk) else "·"
                cols.append(f"{j}:{s}/{m}")
        print(f"{row[1]:<28} {len(sc)}/{len(mk):<5} " + "  ".join(cols) + f"    adv={[round(v,2) for v in token_adv[i].tolist()]}")
    print()

print("场景 A: target=<a_1><b_2><c_3><d_4> (4级)  |  num_generations=4")
run_pipeline(
    [("p_t4", "<a_1><b_2><c_3><d_4>"), ("p_t4", "<a_1><b_2><c_9><d_9>"),
     ("p_t4", "<a_1><b_2><c_3>"), ("p_t4", "<a_1><b_9>")],
    completion_len=5, eos_list=[4, 4, 3, 2],
)
print("场景 B: target=<a_1><b_2><c_3> (3级)  |  num_generations=4")
run_pipeline(
    [("p_t3", "<a_1><b_2><c_3><d_4>"), ("p_t3", "<a_1><b_2><c_3>"),
     ("p_t3", "<a_1><b_9><c_3>"), ("p_t3", "<a_1><b_2><c_9>")],
    completion_len=5, eos_list=[4, 3, 3, 3],
)

print("""设计期望表（分歧位 -1；分歧之后的后缀与 EOS 位 0）：
  精确命中               : 全位 +1（含 EOS）
  中段分歧+继续(a b c9 d9): a+1  b+1  c9-1  d9=0  EOS=0
  提前停止(a b c vs 4级)  : a+1  b+1  c+1         EOS=-1
  多出一级(a b c d vs 3级): a+1  b+1  c+1  d-1   EOS=0
  位置1分歧即停(a b9)     : a+1  b9-1            EOS=0
  位置1分歧继续(a b9 c)   : a+1  b9-1  c=0       EOS=0
  末位分歧(a b c9 vs 3级) : a+1  b+1  c9-1       EOS=0""")
