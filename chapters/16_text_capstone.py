"""Chapter 16 (进阶篇毕业项目): 文本情感分类 —— 把进阶五课拧成一股绳.

这是进阶篇的综合项目。它没有任何"新知识"，考的是把五课拧成一股绳的能力：

    进阶 11 tf.data        ──► 数据管道（加载 / 打乱 / 组批 / 预取）
    进阶 12 序列思维        ──► 整数序列 → 定长窗口的文本张量
    进阶 13 注意力          ──► 自写 SelfAttention 做分类头
    进阶 15 自定义训练      ──► GradientTape + tf.function 手写循环
    第 8 章 保存导出        ──► .keras 交付
    附录 H 评估指标         ──► accuracy + F1 + 混淆矩阵（二分类）

任务：IMDB 影评情感二分类（数据已缓存，无需下载）。纯 CPU 约 3-5 分钟。

After reading this chapter you should be able to:
    * 独立完成"管道 → 模型 → 自定义训练 → 评估 → 导出"全流程，
    * 说出每一环节分别来自哪一章、为什么需要它，
    * 解释为什么 F1 比 accuracy 更能反映这个任务的真实水平。
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

from tf2tutorial.config import FIGURES_DIR, MODELS_DIR, ensure_dirs  # noqa: E402
from tf2tutorial.utils import save_json, set_global_seed  # noqa: E402

VOCAB = 20_000
MAXLEN = 200
BATCH = 128
EMBED_DIM = 32
EPOCHS = 3


def load_sequences() -> tuple:
    """IMDB 整数序列（keras 已缓存）。返回 4 份切好的 numpy 数组。"""
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.imdb.load_data(num_words=VOCAB)
    x_tr = np.asarray(x_tr, dtype=object)
    y_tr = np.asarray(y_tr, dtype=np.float32)

    def pad(seqs):
        out = np.zeros((len(seqs), MAXLEN), dtype=np.int32)  # 0 = <PAD>
        for i, s in enumerate(seqs):
            s = np.asarray(s, dtype=np.int32)[:MAXLEN]
            out[i, : len(s)] = s
        return out

    return pad(x_tr), y_tr, pad(x_te), np.asarray(y_te, dtype=np.float32)


def make_pipeline(x: np.ndarray, y: np.ndarray, shuffle: bool) -> tf.data.Dataset:
    """进阶 11 的标准五算子顺序，一个不少。"""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=42)
    return ds.batch(BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


class SelfAttentionBlock(tf.keras.layers.Layer):
    """进阶 13 的手写注意力（单头版，教学优先）。mask 掉 <PAD> 位置的打分。"""

    def __init__(self, d: int = EMBED_DIM, **kw):
        super().__init__(**kw)
        self.d = d
        self.wq = tf.keras.layers.Dense(d)
        self.wk = tf.keras.layers.Dense(d)
        self.wv = tf.keras.layers.Dense(d)

    def call(self, x, mask):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(
            tf.cast(self.d, tf.float32))
        pad_mask = tf.cast(mask[:, None, :], tf.float32)  # (B, 1, T)
        scores = scores + (1.0 - pad_mask) * (-1e9)       # PAD 位置打分为 -inf
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, v)


def build_model() -> tf.keras.Model:
    # 注意: Keras 3 的函数式布线要用 keras.ops（tf.* 直接作用在 KerasTensor 上会报错）；
    # 层的 call() 内部可以用 tf.*（构建时以真实张量追踪）。这也是附录 D 的一个知识点。
    ops = tf.keras.ops
    inp = tf.keras.layers.Input(shape=(MAXLEN,), dtype=tf.int32)
    pad_mask = ops.not_equal(inp, 0)                      # 0 = <PAD>
    h = tf.keras.layers.Embedding(VOCAB, EMBED_DIM)(inp)
    h = SelfAttentionBlock()(h, pad_mask)
    # 掩码平均池化：只对真实 token 取平均（而不是把 PAD 也平均进去）
    mask = ops.cast(ops.expand_dims(pad_mask, axis=-1), "float32")
    h = ops.sum(h * mask, axis=1) / (ops.sum(mask, axis=1) + 1e-9)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(h)
    return tf.keras.Model(inp, out, name="text_capstone")


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """附录 H 的 F1：精确率与召回率的调和平均（二分类，阈值 0.5）。"""
    pred = (y_pred >= 0.5).astype(np.float32)
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return 2 * precision * recall / (precision + recall + 1e-9)


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 16 (进阶篇毕业项目) — 文本情感分类：五课合一")
    x_tr, y_tr, x_te, y_te = load_sequences()
    n_val = 5000
    (x_va, y_va), (x_tr, y_tr) = (x_tr[:n_val], y_tr[:n_val]), (x_tr[n_val:], y_tr[n_val:])
    print(f"IMDB: 训练 {len(x_tr)} / 验证 {len(x_va)} / 测试 {len(x_te)}，"
          f"定长 {MAXLEN}（管道见进阶 11）\n")

    train_ds = make_pipeline(x_tr, y_tr, shuffle=True)
    val_ds = make_pipeline(x_va, y_va, shuffle=False)
    test_ds = make_pipeline(x_te, y_te, shuffle=False)

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    bce = tf.keras.losses.BinaryCrossentropy()

    @tf.function  # 进阶 15：整步下沉到图模式
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            pred = model(xb, training=True)
            loss = bce(yb, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print("--- 自定义训练循环（进阶 15）---")
    history = {"loss": [], "val_acc": []}
    for epoch in range(EPOCHS):
        losses = []
        for xb, yb in train_ds:
            losses.append(float(train_step(xb, yb)))
        # 验证集准确率
        correct = total = 0
        for xb, yb in val_ds:
            pred = model(xb, training=False)
            correct += int(np.sum((pred.numpy().flatten() >= 0.5) == (yb.numpy() == 1)))
            total += len(yb)
        val_acc = correct / total
        history["loss"].append(float(np.mean(losses)))
        history["val_acc"].append(val_acc)
        print(f"  epoch {epoch + 1}: loss={np.mean(losses):.4f} val_acc={val_acc:.4f}")

    # ---- 评估（附录 H）：accuracy 只是开始，F1 才见真章 ----
    y_true_all, y_pred_all = [], []
    for xb, yb in test_ds:
        pred = model(xb, training=False).numpy().flatten()
        y_true_all.append(yb.numpy())
        y_pred_all.append(pred)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    acc = float(np.mean((y_pred_all >= 0.5) == (y_true_all == 1)))
    f1 = f1_score(y_true_all, y_pred_all)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true_all.astype(int), (y_pred_all >= 0.5).astype(int)):
        cm[t, p] += 1
    print(f"\n测试集: accuracy={acc:.4f}  F1={f1:.4f}")
    print(f"混淆矩阵 [[TN FP] [FN TP]]:\n{cm}")
    print("为什么还要看 F1/混淆矩阵？acc 只告诉你总体对错，")
    print("不告诉你'把差评误判成好评'和'把好评误判成差评'哪个更严重。")

    # ---- 导出（第 8 章）：交付 .keras 整模型 ----
    out_keras = MODELS_DIR / "chapter16_text_capstone.keras"
    model.save(out_keras)
    print(f"\n模型已交付: {out_keras.name}")

    # ---- 图 ----
    P = "#5B21B6"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(history["loss"], "o-", color=P)
    axes[0].set_title("train loss"); axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_acc"], "o-", color="#059669")
    axes[1].set_title("val accuracy"); axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.5, 1.0); axes[1].grid(alpha=0.3)
    fig.suptitle("chapter 16 — capstone training")
    fig.tight_layout()
    out1 = FIGURES_DIR / "chapter16_history.png"
    fig.savefig(out1, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["pred 负", "pred 正"])
    ax.set_yticks([0, 1], ["true 负", "true 正"])
    ax.set_title(f"chapter 16 — confusion (acc={acc:.3f}, F1={f1:.3f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out2 = FIGURES_DIR / "chapter16_confusion.png"
    fig.savefig(out2, dpi=120); plt.close(fig)
    save_json({"accuracy": acc, "f1": f1, "epochs": EPOCHS,
               "confusion": cm.tolist()}, FIGURES_DIR / "chapter16_summary.json")
    print(f"figures saved: {out1}, {out2}")

    print("\n--- 动手试一试 ---")
    print("  1. 把 maxlen 从 200 改成 80，比较 F1 —— 截断长影评丢了什么信息？")
    print("  2. 把 SelfAttentionBlock 换成第 5 章的词袋 + Dense，F1 差多少？")
    print("     （这是'注意力到底带来了什么'的定量答案）")
    print("  3. 给训练循环加早停（进阶 07 的逻辑自己用 tape 实现）——")
    print("     没有 fit 的 callbacks 可用，你会发现'回调'不过是每轮末尾的几行 if。")

    print("\nKey takeaways:")
    print("  * 管道(11) + 序列(12) + 注意力(13) + 自定义训练(15) + 导出(08) 全流程打通")
    print("  * mask 是序列模型的地基：注意力打分和池化都必须'看不见 PAD'")
    print("  * 评估看 F1 和混淆矩阵，不只是 accuracy（附录 H）")
    print("  * 恭喜——进阶篇毕业。去 study_plan_semester.md 选你的下一条路")


if __name__ == "__main__":
    main()
