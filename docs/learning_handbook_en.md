# Learning Handbook: TensorFlow 2 from Zero

> Project: `F:\Tensorflow` (`tf2-tutorial`)
> Audience: engineers / students who have Python installed and want to learn deep-learning basics by running code.

---

## 0. Roadmap

```
chapters/01  ──► chapters/02  ──► chapters/03  ──► chapters/04
   |              |              |              |
 NumPy         Keras          MLP on         CNN on
 autograd      Sequential     MNIST          CIFAR-10
   |
   +──► chapters/05 (IMDB) ──► chapters/06 (Transfer Learning)
        chapters/07 (Callbacks / TensorBoard) ──► chapters/08 (Save / Export)
```

- Chapter 1 uses only NumPy on purpose so it works everywhere.
- Chapters 2–8 layer on concepts. Each chapter ends with a saved training-curve PNG.
- `src/tf2tutorial/` is the reusable library; chapter scripts are the consumer.

### 0.1 Local environment

TensorFlow 2.21 lives inside `F:\tf2-env` (Python 3.12, CPU only). The recommended
way to launch any chapter is through the dispatcher script:

```powershell
python scripts\use_tf2_env.py chapters\01_tensors_autograd.py
python scripts\use_tf2_env.py chapters\03_mlp_mnist.py
```

`scripts\use_tf2_env.py` calls `F:\tf2-env\Scripts\python.exe` for you, so there
is no need to activate the venv first.

---

## 1. Chapter 01 — Tensors & "autograd" demystified

Goal: understand (a) what a tensor is, (b) the forward / loss / backward / update loop.

Entry point: `chapters/01_tensors_autograd.py`.

Key ideas:

1. **Tensor = N-D array**. `np.array` and `tf.constant` behave almost the same.
2. **Broadcasting**: `(2,3) + (1,3)` silently expands the second term — no Python loops.
3. **Matmul**: `a @ a.T` is the heart of every fully-connected layer.
4. **Hand-rolled gradient**: derive `(sigmoid(w*x + b) - y)^2` w.r.t. `w`, `b`. In TensorFlow this is exactly what `tf.GradientTape` automates.

Run:

```powershell
python chapters/01_tensors_autograd.py
```

Inspect `figures/chapter01_loss_surface.png` to see the minimum of the bowl.

---

## 2. Chapter 02 — first Keras model

```python
from tensorflow.keras import layers, models
model = models.Sequential([layers.Input(shape=(1,)), layers.Dense(1)])
model.compile(optimizer="sgd", loss="mse")
model.fit(x, y, epochs=30)
```

- `input_shape` is the per-sample shape — never include the batch dim.
- Regression: leave the last layer without an activation so the network learns a linear map.

---

## 3. Chapter 03 — MLP on MNIST

Reuse `build_mlp` from `src/tf2tutorial/models.py`. The teaching points:

- reshape `28×28` → `784` and normalize to `[0,1]`;
- `validation_split=0.1` only takes a slice of training data; the test set stays independent;
- The final figure is saved under `figures/chapter03_history.png`.

---

## 4. Chapter 04 — CNN on CIFAR-10

`Conv2D` + `MaxPooling2D` + `Dropout`. Why does it outperform an MLP?

- **Local receptive fields** match natural-image statistics.
- **Weight sharing** keeps parameter count manageable.
- **Pooling** gives translation invariance.

CPU tip: on this machine five epochs took about 93 seconds and reached 72.6%
test accuracy, after a one-time ~170 MB CIFAR-10 download. Start with
`EPOCHS=1` to validate the pipeline, then bump it.

---

## 5. Chapter 05 — IMDB text classification

```
integer tokens → one-hot (10,000) → Dense → sigmoid
```

The one-hot baseline is deliberately simple. As an exercise, swap in the
Embedding-based `build_text_classifier` from `src/tf2tutorial/models.py` and
compare accuracy and speed.

---

## 6. Chapter 06 — Transfer learning

```python
base = MobileNetV2(weights="imagenet", include_top=False)
base.trainable = False           # Phase 1: train only the head
# ... train ...
base.trainable = True            # Phase 2: fine-tune last 20 layers
for layer in base.layers[:-20]:
    layer.trainable = False
```

---

## 7. Chapter 07 — Callbacks & TensorBoard

- `TensorBoard`: live scalar / histogram views.
- `ModelCheckpoint`: keep only the best epoch.
- `EarlyStopping`: avoid wasting compute when val_loss plateaus.
- `LambdaCallback`: hook in any custom logic.

```powershell
tensorboard --logdir F:\Tensorflow\output\logs
```

---

## 8. Chapter 08 — Save & export

| Format | Strength | Typical use |
| --- | --- | --- |
| `.keras` | Single file, contains graph + weights + optimizer state | Distribute to teammates |
| `SavedModel` | Standard for TF Serving / TFLite conversion | Server-side serving |
| `.tflite` | Compressed, mobile / edge friendly | On-device inference |

Artifacts: `models/chapter08_mnist_mlp.keras`, `models/saved/chapter08_mnist_mlp/`
(SavedModel), and `models/chapter08_mnist_mlp.tflite`. Export uses Keras 3
`model.export(...)` (TensorFlow ≥ 2.16) with a `tf.saved_model.save` fallback.

---

## 9. Quick troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `ModuleNotFoundError: tensorflow` | Not installed: `pip install tensorflow` |
| Accuracy stuck at 1/N | Wrong loss for your label format (sparse vs. one-hot) |
| OOM | Reduce `BATCH_SIZE` / `IMAGE_SIZE` |
| Training is very slow | Default config is CPU; drop `EPOCHS` for the slow chapters |

---

## 10. Further reading

- Chollet, *Deep Learning with Python* (2nd) chapters 4–7.
- Géron, *Hands-On ML* (3rd) chapter 10.
- https://www.tensorflow.org/tutorials
