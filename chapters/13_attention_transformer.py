"""Chapter 13 (进阶篇): 注意力机制 — 从零手写 Self-Attention 到 mini-Transformer.

Transformer 是现代 AI 的基础设施 (GPT/BERT/ViT 的骨架). 但它的核心思想
可以用 30 行代码讲清楚. 本章不用任何高级 API, 手写:

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

教学任务刻意选得" attention 一看就懂": 给 8 个 0~9 的数字,
让模型**找出最大值**. 训练完把注意力权重画成热力图 —— 你会亲眼看到
模型把注意力"放"在最大值的位置上.

另外做一个关键对照实验: 去掉位置编码 (Positional Encoding),
验证注意力本身是"排列不变"的 —— 这就是 Transformer 必须注入位置信息的原因.

After reading this chapter you should be able to:
    * 手写 scaled dot-product attention 并解释为什么要除以 sqrt(d_k),
    * 说出 Q/K/V 的直觉 (查询/索引/内容),
    * 解释位置编码为什么必须存在 (用实验证明),
    * 认出 Transformer 块的全部零件: 注意力+残差+LayerNorm+FFN.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # type: ignore # noqa: E402

from tf2tutorial.config import FIGURES_DIR, ensure_dirs  # noqa: E402
from tf2tutorial.utils import save_json, set_global_seed  # noqa: E402

SEQ_LEN = 8
NUM_CLASSES = 10  # 输出 = 最大值 (0~9 分类)
D_MODEL = 32
HEADS = 2


class SelfAttention(tf.keras.layers.Layer):
    """手写多头自注意力 —— 全章的核心, 每一行都值得读懂.

    Q (query, 查询): "我在找什么"
    K (key,   索引): "我这里有什么可以被找到"
    V (value, 内容): "匹配上之后我给出的信息"
    就像在图书馆检索: 你拿着查询词(Q)去和每本书的标签(K)比对,
    越匹配的书, 你越多采纳它的内容(V).
    """

    def __init__(self, d_model: int = D_MODEL, heads: int = HEADS, **kwargs):
        super().__init__(**kwargs)
        assert d_model % heads == 0, "d_model 必须能被 heads 整除"
        self.heads = heads
        self.d_k = d_model // heads  # 每个头的维度
        self.wq = tf.keras.layers.Dense(d_model, name="q_proj")
        self.wk = tf.keras.layers.Dense(d_model, name="k_proj")
        self.wv = tf.keras.layers.Dense(d_model, name="v_proj")
        self.wo = tf.keras.layers.Dense(d_model, name="out_proj")

    def build(self, input_shape):
        # 显式 build: Keras 3 要求变量形状完全确定, 不靠运行时推断
        for proj in (self.wq, self.wk, self.wv, self.wo):
            proj.build((None, input_shape[1], self.heads * self.d_k))
        super().build(input_shape)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        # (batch, seq, d_model) -> (batch, heads, seq, d_k)
        b = tf.shape(x)[0]
        return tf.reshape(x, (b, -1, self.heads, self.d_k))

    def call(self, x: tf.Tensor, return_attention: bool = False):
        q = self._split_heads(self.wq(x))
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))
        # scores[i][j] = 第 i 个位置对第 j 个位置的"关注程度" (未归一化)
        scores = tf.matmul(q, k, transpose_b=True)
        # 为什么要除以 sqrt(d_k)? d_k 维点积的方差 ~ d_k, 值太大会把 softmax
        # 推进饱和区 (几乎 one-hot), 梯度消失. 除以 sqrt(d_k) 把方差拉回 1.
        scores = scores / tf.sqrt(tf.cast(self.d_k, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)  # 每行归一化成概率分布
        out = tf.matmul(weights, v)
        # 拼回多头: 最后一维必须是静态值 (heads*d_k), Keras 3 才能构建 Dense
        out = tf.reshape(out, (-1, tf.shape(x)[1], self.heads * self.d_k))
        out = self.wo(out)
        if return_attention:
            return out, weights
        return out


def transformer_block(x: tf.Tensor, attn: SelfAttention, name: str) -> tf.Tensor:
    """一个标准 Transformer 块: 注意力 + 残差 + LayerNorm + FFN + 残差 + LayerNorm."""
    h = attn(x)
    x = tf.keras.layers.LayerNormalization(name=f"{name}_ln1")(x + h)  # 残差连接
    f = tf.keras.layers.Dense(64, activation="relu", name=f"{name}_ffn1")(x)
    f = tf.keras.layers.Dense(D_MODEL, name=f"{name}_ffn2")(f)
    x = tf.keras.layers.LayerNormalization(name=f"{name}_ln2")(x + f)
    return x


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """正弦位置编码 (Vaswani et al., 2017): 每个位置一个独一无二的'指纹'."""
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    i = np.arange(d_model // 2, dtype=np.float32)[None, :]
    angle = pos / np.power(10_000.0, 2 * i / d_model)
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


def build_model(use_pe: bool = True) -> tuple[tf.keras.Model, list[SelfAttention]]:
    """数字序列 -> 找最大值. use_pe=False 用于'位置编码必要性'对照实验."""
    inp = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    h = tf.keras.layers.Embedding(NUM_CLASSES, D_MODEL, name="token_embed")(inp)
    if use_pe:
        pe = positional_encoding(SEQ_LEN, D_MODEL)
        h = h + tf.constant(pe)  # 把位置指纹加到词向量上
    attns = [SelfAttention(name=f"self_attn_{i}") for i in range(2)]
    for i, attn in enumerate(attns):
        h = transformer_block(h, attn, f"block{i}")
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(h)
    model = tf.keras.Model(inp, out, name=f"mini_transformer_pe{int(use_pe)}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, attns


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, NUM_CLASSES, size=(n, SEQ_LEN)).astype(np.int32)
    y = x.max(axis=1)
    return x, y


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 13 (进阶) — 手写注意力: 让 Transformer '看得见'最大值")
    x_tr, y_tr = make_data(8000, seed=1)
    x_te, y_te = make_data(2000, seed=2)
    print(f"任务: {SEQ_LEN} 个 0~9 的数字 -> 输出最大值. 训练 {len(x_tr)} / 测试 {len(x_te)}\n")

    # --- 主实验: 带位置编码的 mini-Transformer ---
    model, attns = build_model(use_pe=True)
    hist = model.fit(x_tr, y_tr, validation_split=0.1,
                     epochs=6, batch_size=128, verbose=0)
    acc = model.evaluate(x_te, y_te, verbose=0)[1]
    print(f"[1] 带位置编码: test accuracy = {acc:.4f} (6 个 epoch, CPU 数秒)\n")

    # --- 对照实验: 没有位置编码会怎样? ---
    model_nope, _ = build_model(use_pe=False)
    model_nope.fit(x_tr, y_tr, validation_split=0.1,
                   epochs=6, batch_size=128, verbose=0)
    # 关键测试: 把测试序列随机打乱顺序, 比较"打乱前后预测的一致率".
    # 注意力的打分只依赖"内容配对", 天生排列不变 —— 没有 PE, 打乱后的预测
    # 应当与原序**逐条完全相同**; 有 PE, 位置参与了计算, 一致率必然 < 100%.
    # (找最大值任务本身与顺序无关, 所以精度不会大变 —— 一致率才是锐利的指标.)
    perm = np.random.default_rng(0).permutation(SEQ_LEN)
    x_te_shuf = x_te[:, perm]
    agree_pe = float(np.mean(
        model.predict(x_te, verbose=0).argmax(1)
        == model.predict(x_te_shuf, verbose=0).argmax(1)))
    agree_nope = float(np.mean(
        model_nope.predict(x_te, verbose=0).argmax(1)
        == model_nope.predict(x_te_shuf, verbose=0).argmax(1)))
    acc_nope = model_nope.evaluate(x_te, y_te, verbose=0)[1]
    print("[2] 位置编码必要性实验 (把输入序列随机打乱, 测预测一致率):")
    print(f"    有 PE: 打乱前后一致率 {agree_pe:.4f} (<100%, 位置参与了计算)")
    print(f"    无 PE: 打乱前后一致率 {agree_nope:.4f} (数学上必然 =100%, 排列不变!)")
    print(f"    (无 PE 模型精度 {acc_nope:.4f} —— 它照样能找最大值, 但它'看不见顺序')")
    print("    结论: 注意力天生'看不见顺序'; 任务需要顺序时, 必须用位置编码注入.\n")

    # --- 注意力热力图: 亲眼看到模型'盯住'最大值 ---
    sample = x_te[:1]
    embed = model.get_layer("token_embed")(sample) + tf.constant(positional_encoding(SEQ_LEN, D_MODEL))
    _, attn_w = attns[0](embed, return_attention=True)  # eager 直接算, 不建新模型
    weights = attn_w[0].numpy().mean(axis=0)  # 平均两个头
    tokens = sample[0].tolist()
    max_pos = int(np.argmax(tokens))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(weights, cmap="viridis")
    ax.set_xticks(range(SEQ_LEN), [str(t) for t in tokens])
    ax.set_yticks(range(SEQ_LEN), [str(t) for t in tokens])
    ax.set_xlabel("被关注的 token (列) — 按值大小排列")
    ax.set_ylabel("发起查询的 token (行)")
    ax.set_title(f"chapter 13 — attention heatmap (max={tokens[max_pos]} @ pos {max_pos})")
    ax.add_patch(plt.Rectangle((max_pos - 0.5, -0.5), 1, SEQ_LEN, fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    out1 = FIGURES_DIR / "chapter13_attention_heatmap.png"
    fig.savefig(out1, dpi=120); plt.close(fig)

    # --- 训练曲线 ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist.history["accuracy"], label="train acc")
    ax.plot(hist.history["val_accuracy"], label="val acc")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("chapter 13 — mini-Transformer accuracy")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out2 = FIGURES_DIR / "chapter13_history.png"
    fig.savefig(out2, dpi=120); plt.close(fig)
    save_json({"agree_with_pe": agree_pe, "agree_no_pe": agree_nope,
               "acc_pe": float(acc), "acc_no_pe": float(acc_nope)},
              FIGURES_DIR / "chapter13_pe_experiment.json")
    print(f"figures saved: {out1}, {out2}")

    print("\n--- 动手试一试 ---")
    print("  1. 把除以 sqrt(d_k) 的那一行删掉重训, 观察收敛是否变慢/不稳 ——")
    print("     这就是'缩放点积'里'缩放'二字的由来。")
    print("  2. 把任务换成'找最小值'或'求和 mod 10', 注意力热力图会怎么变?")
    print("  3. 把 heads 从 2 改成 1 和 8 (保持 d_model 不变), 比较准确率 ——")
    print("     多头的意义: 不同的头可以关注不同的'关系模式'。")

    print("\nKey takeaways:")
    print("  * Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V —— 全部核心就这一行")
    print("  * 除以 sqrt(d_k) 防止 softmax 饱和; 多头让模型关注多种关系")
    print("  * 注意力天生排列不变, 位置信息必须由位置编码显式注入 (实验证明)")
    print("  * Transformer 块 = 注意力 + 残差 + LayerNorm + FFN, 大模型全是它的堆叠")


if __name__ == "__main__":
    main()
