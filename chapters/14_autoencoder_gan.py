"""Chapter 14 (进阶篇): 生成模型入门 — 自编码器 (AE) 与生成对抗网络 (GAN).

前面所有章节都是"判别式"任务: 给 x, 预测 y. 本章换一个视角 ——
让模型学习数据本身的分布, **无中生有**:

    自编码器 (AE): 压缩 -> 重建. 学会"什么信息是本质的" (无监督表征学习)
    生成对抗 (GAN): 生成器造假 vs 判别器打假, 博弈中共同进化

两个模型都在 MNIST 上用全连接网络实现, 纯 CPU 几分钟跑完.
AE 重建会相当清晰; GAN 只训几分钟, 得到的是"雏形数字" ——
卷积 + 长时间训练才能到以假乱真 (诚实预期!).

After reading this chapter you should be able to:
    * 解释 AE 的"瓶颈"为什么逼模型学出压缩表征,
    * 手写 GAN 的交替训练循环 (GradientTape 实战),
    * 读懂 GAN 的 loss 曲线为什么震荡 (不收敛是常态),
    * 说出 mode collapse 是什么、为什么会发生。
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

LATENT_DIM = 32
IMG = 28 * 28


def load_mnist_flat() -> np.ndarray:
    (x, _), _ = tf.keras.datasets.mnist.load_data()
    x = x.astype(np.float32) / 255.0  # 归一化到 [0,1] (sigmoid 输出的值域)
    return x.reshape(-1, IMG)


def show_grid(images: np.ndarray, title: str, path: Path, n: int = 8) -> None:
    """把 n*n 张 28x28 图拼成网格保存."""
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_autoencoder() -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """AE: 784 -> 128 -> 32 (瓶颈) -> 128 -> 784.

    瓶颈的 32 维逼网络扔掉细节、保留"画的是哪个数字"的本质信息.
    这就是无监督表征学习的雏形 —— BERT/GPT 的预训练也是"从数据本身学习".
    """
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(LATENT_DIM, activation="relu", name="bottleneck"),
    ], name="encoder")
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LATENT_DIM,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(IMG, activation="sigmoid"),  # 输出 [0,1] 像素
    ], name="decoder")
    ae_inp = tf.keras.Input(shape=(IMG,), name="ae_input")
    ae = tf.keras.Model(ae_inp, decoder(encoder(ae_inp)), name="autoencoder")
    ae.compile(optimizer="adam", loss="mse")
    return ae, encoder, decoder


def build_generator() -> tf.keras.Model:
    """生成器: 64 维噪声 -> 784 维"手写数字". 随机噪声就是它的'想象力来源'."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dense(128, activation="leaky_relu"),
        tf.keras.layers.Dense(256, activation="leaky_relu"),
        tf.keras.layers.Dense(IMG, activation="sigmoid"),  # 与 AE 一致, [0,1] 域
    ], name="generator")


def build_discriminator() -> tf.keras.Model:
    """判别器: 一张图 -> "真/假"概率. 它就是第 5 章的二分类器!"""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG,)),
        tf.keras.layers.Dense(256, activation="leaky_relu"),
        tf.keras.layers.Dense(128, activation="leaky_relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="discriminator")


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 14 (进阶) — 生成模型入门: 自编码器与 GAN")
    x = load_mnist_flat()
    print(f"MNIST: {len(x)} 张 28x28, 已归一化到 [0,1]\n")

    # ================= part 1: 自编码器 =================
    print("--- Part 1: 自编码器 (压缩 -> 重建) ---")
    ae, encoder, decoder = build_autoencoder()
    ae.fit(x, x, epochs=3, batch_size=256, verbose=0)  # 输入=输出: 学"恒等", 但瓶颈逼它压缩
    recon = ae.predict(x[:64], verbose=0)
    mse_ae = float(np.mean((x[:64] - recon) ** 2))
    print(f"重建 MSE = {mse_ae:.5f} (784 维 -> 32 维 -> 784 维, 信息只剩 4%)")
    show_grid(x[:64], "chapter 14 — AE originals (top row grid)",
              FIGURES_DIR / "chapter14_ae_originals.png")
    show_grid(recon, "chapter 14 — AE reconstructions (from 32-dim bottleneck)",
              FIGURES_DIR / "chapter14_ae_reconstructions.png")

    # 潜在空间的算术: 相似的数字, 潜在向量也相似吗?
    # 取两个样本的 latent 做线性插值, 逐帧解码 —— 如果过渡平滑, 说明空间有结构.
    lat = encoder(x[:2], training=False).numpy()  # (2, 32) 两个样本的潜在向量
    alphas = np.linspace(0, 1, 10, dtype=np.float32).reshape(-1, 1)
    mix = lat[0][None, :] * alphas + lat[1][None, :] * (1.0 - alphas)  # (10, 32)
    interp = decoder(mix, training=False)
    fig, axes = plt.subplots(1, 10, figsize=(10, 1.6))
    for i, ax in enumerate(axes):
        ax.imshow(interp[i].numpy().reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{alphas[i, 0]:.1f}", fontsize=7)
        ax.axis("off")
    fig.suptitle("latent interpolation: sample0 -> sample1 (smooth = structured space)")
    fig.tight_layout()
    out3 = FIGURES_DIR / "chapter14_ae_interpolation.png"
    fig.savefig(out3, dpi=120); plt.close(fig)

    # ================= part 2: GAN =================
    print("\n--- Part 2: GAN (生成器 vs 判别器的博弈) ---")
    g = build_generator()
    d = build_discriminator()
    bce = tf.keras.losses.BinaryCrossentropy()
    opt_g = tf.keras.optimizers.Adam(2e-4)
    opt_d = tf.keras.optimizers.Adam(2e-4)
    ds = tf.data.Dataset.from_tensor_slices(x).shuffle(10_000, seed=42).batch(64, drop_remainder=True)

    @tf.function
    def train_step(real: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        noise = tf.random.normal((64, 64))
        fake = g(noise, training=True)
        # -- 判别器: 真图标 1, 假图标 0 (注意 label smoothing: 真图标 0.9 防止过自信) --
        with tf.GradientTape() as tape_d:
            d_real = d(real, training=True)
            d_fake = d(fake, training=True)
            loss_d = bce(tf.ones_like(d_real) * 0.9, d_real) + bce(tf.zeros_like(d_fake), d_fake)
        grad_d = tape_d.gradient(loss_d, d.trainable_variables)
        opt_d.apply_gradients(zip(grad_d, d.trainable_variables))
        # -- 生成器: 让判别器把假图看成真的 (label=1) --
        noise = tf.random.normal((64, 64))
        with tf.GradientTape() as tape_g:
            d_fake = d(g(noise, training=True), training=True)
            loss_g = bce(tf.ones_like(d_fake), d_fake)
        grad_g = tape_g.gradient(loss_g, g.trainable_variables)
        opt_g.apply_gradients(zip(grad_g, g.trainable_variables))
        return loss_g, loss_d

    g_losses, d_losses = [], []
    steps = 3000
    for step in range(steps):
        for real_batch in ds.take(1):  # 每个 step 用一个 batch
            lg, ld = train_step(real_batch)
        g_losses.append(float(lg))
        d_losses.append(float(ld))
        if (step + 1) % 1000 == 0:
            print(f"  step {step + 1:5d}: loss_g={lg:.3f} loss_d={ld:.3f}")

    # 生成样本 + loss 曲线
    show_grid(g(tf.random.normal((64, 64)), training=False).numpy(),
              "chapter 14 — GAN samples (3k steps, dense net — honest expectation: blobby digits)",
              FIGURES_DIR / "chapter14_gan_samples.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g_losses, label="generator loss", alpha=0.7)
    ax.plot(d_losses, label="discriminator loss", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("chapter 14 — GAN losses (oscillation is NORMAL, not a bug)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out2 = FIGURES_DIR / "chapter14_gan_losses.png"
    fig.savefig(out2, dpi=120); plt.close(fig)
    save_json({"ae_mse": mse_ae, "gan_steps": steps,
               "g_loss_final": g_losses[-1], "d_loss_final": d_losses[-1]},
              FIGURES_DIR / "chapter14_summary.json")
    print(f"\nfigures saved: {FIGURES_DIR / 'chapter14_ae_*.png'}, {out2}, {out3}")
    print("GAN loss 曲线为什么震荡? 因为它在解一个极小极大博弈而不是下降问题:")
    print("G 变强 -> D 掉点 -> D 反扑 -> G 掉点 —— 双方都在'追'对方, 不存在共同的谷底.")

    print("\n--- 动手试一试 ---")
    print("  1. 把 LATENT_DIM 从 32 改成 4, 重建质量怎么变? 潜在空间越窄, 表征越'本质'也越粗糙。")
    print("  2. 把判别器 learning rate 乘 10 —— 弱势的 G 会被带向哪? 观察样本质量的崩塌")
    print("     (训练失衡是 GAN 最常见的翻车方式)。")
    print("  3. 只用数字 '1' 的样本训练 AE, 再输入一张 '7' —— 重建结果像什么?")
    print("     (这就是'分布外'输入: 模型只能用它见过的分布去解释世界)")

    print("\nKey takeaways:")
    print("  * AE 用瓶颈逼模型学压缩表征 = 无监督表征学习的雏形")
    print("  * GAN = 极小极大博弈: loss 震荡是常态, '收敛'的含义与监督学习不同")
    print("  * 训练失衡 (一方过强) 是 GAN 翻车的头号原因; label smoothing 是常用缓冲")
    print("  * 生成模型谱系: AE -> VAE -> GAN -> Diffusion (后者是当前主流)")


if __name__ == "__main__":
    main()
