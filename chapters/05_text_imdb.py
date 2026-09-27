"""Chapter 05: text classification on IMDB (positive / negative reviews).

Concepts:
    * integer-encoded text -> one-hot bag-of-words,
    * Dense-only classifier as a baseline,
    * an Embedding-based alternative lives in ``src/tf2tutorial/models.py``
      (``build_text_classifier``) — try it as an exercise and compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from tf2tutorial.config import FIGURES_DIR, ensure_dirs
from tf2tutorial.data import load_imdb, vectorize_sequences
from tf2tutorial.training import history_to_dict, save_history
from tf2tutorial.utils import set_global_seed, timer
from tf2tutorial.visualize import plot_history

VOCAB_SIZE = 10_000  # 为什么限制词表大小为 1 万？
# 词表越大，输入向量维度越高，参数越多，越容易过拟合。
# 而且生僻词（排名 1 万以后的）出现频率很低，对情感分类的贡献不大。


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 05 — IMDB sentiment classification")
    x_train, y_train, x_test, y_test = load_imdb(num_words=VOCAB_SIZE)

    # 为什么要 vectorize（向量化）？神经网络只能处理数字，不能直接吃文字。
    # 这里用的是"词袋"方法：每句话变成一个 10000 维的 0/1 向量，
    # 第 i 位是 1 表示这句话里出现了词表中第 i 个词。
    x_train = vectorize_sequences(x_train, dimension=VOCAB_SIZE)
    x_test = vectorize_sequences(x_test, dimension=VOCAB_SIZE)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    x_val = x_train[:10_000]
    y_val = y_train[:10_000]
    x_train = x_train[10_000:]
    y_train = y_train[10_000:]

    from tensorflow.keras import layers, models  # type: ignore
    from tensorflow.keras import losses, optimizers, metrics  # type: ignore

    model = models.Sequential([
        layers.Input(shape=(VOCAB_SIZE,)),
        # 为什么隐藏层只用 16 个神经元？输入是 10000 维，如果隐藏层太大，
        # 参数量会爆炸，而且词袋向量很稀疏，模型非常容易过拟合。小网络反而泛化更好。
        layers.Dense(16, activation="relu"),
        # 为什么 Dropout 设 0.5 这么大？文本分类任务中，输入维度高、信息稀疏，
        # 过拟合是主要问题，较大的 Dropout 能有效防止模型"死记硬背"训练数据。
        layers.Dropout(0.5),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.5),
        # 为什么最后一层用 sigmoid 而不是 softmax？因为这是二分类（正面/负面），
        # 只需要一个输出值（0~1 之间）表示"正面的概率"。
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        # 为什么用 RMSprop？词袋向量非常稀疏（大部分是 0），
        # RMSprop 对每个参数有自适应学习率，在稀疏数据上通常比 SGD 效果好。
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        # 为什么用 BinaryCrossentropy？二分类的标准损失，衡量预测概率和真实标签的差距。
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy(name="accuracy")],
    )
    model.summary(print_fn=print)

    with timer("chapter05 fit"):
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=10,
            batch_size=512,
            verbose=2,
        )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\ntest loss: {test_loss:.4f}  test acc: {test_acc:.4f}")

    save_history(history, FIGURES_DIR / "chapter05_history.json")
    plot_history(
        history_to_dict(history),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter05_history.png",
        title="chapter 05 — IMDB",
    )
    print(f"  figure -> {FIGURES_DIR / 'chapter05_history.png'}")

    print("\n--- 动手试一试 ---")
    print("  1. 把隐藏层从 16 改成 64 或 256，看看训练准确率和验证准确率的差距——")
    print("     差距变大了还是变小了？你预计：隐藏层越大越容易过拟合，对吗？")
    print("  2. 注释掉两层 Dropout，再跑一次，看看过拟合程度（训练 acc - 验证 acc）怎么变？")
    print("     你预计会发生什么？没有 Dropout 会严重过拟合。实际结果和预期一致吗？")
    print("  3. 把 VOCAB_SIZE 从 10000 改成 2000 或 20000，对比准确率和过拟合情况。")
    print("     你预计：词表越大参数越多，过拟合越严重？但信息更多可能准确率更高？试试看。")


if __name__ == "__main__":
    main()
