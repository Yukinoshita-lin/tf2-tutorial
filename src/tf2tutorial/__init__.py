"""tf2tutorial: a teaching-oriented TensorFlow 2 toolkit.

Modules:
    config         - path & runtime configuration.
    data           - dataset loaders (MNIST, CIFAR-10, IMDB, tf_flowers).
    models         - reusable model builders (MLP, CNN, text, transfer-learning).
    training       - training loops, callbacks, and evaluation helpers.
    visualize      - matplotlib-based plotting utilities.
    visualize_cnn  - CNN feature visualization (kernels, feature maps, occlusion).
    utils          - small generic helpers (seeding, formatting, IO).
"""

from tf2tutorial import config, data, models, training, utils, visualize, visualize_cnn

__all__ = ["config", "data", "models", "training", "utils", "visualize", "visualize_cnn"]
__version__ = "0.1.0"
