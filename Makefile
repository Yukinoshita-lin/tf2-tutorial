# tf2-tutorial Makefile
# 在 Windows 上推荐使用 Git Bash / WSL；若无 make，可直接运行 python 命令；
# 或者使用 run_all_chapters.py 作为 make chapters 的等价入口。

PY     := python
PIP    := $(PY) -m pip
CHAPTERS := 00_ml_basics 01_tensors_autograd 02_linear_regression 03_mlp_mnist \
            04_cnn_cifar10 05_text_imdb 06_transfer_learning \
            07_callbacks_tensorboard 08_save_and_export \
            09_capstone_image_classifier 10_edge_raspberry_pi

.PHONY: help install install-tf test clean chapter00 chapter01 chapter02 chapter03 \
        chapter04 chapter05 chapter06 chapter07 chapter08 chapter09 chapter10 \
        chapters download notebooks

help:
	@echo "make install     安装基础依赖 (numpy/matplotlib)"
	@echo "make install-tf  安装完整依赖 (含 tensorflow)"
	@echo "make test        跑测试"
	@echo "make chapters    跑全部章节（Windows 等价: python run_all_chapters.py）"
	@echo "make chapterNN   跑指定章节 (NN=00..10)"
	@echo "make notebooks   从 chapters 重新生成 Jupyter 笔记本"
	@echo "make download    下载常用数据集 (MNIST/CIFAR-10/IMDB/tf_flowers)"
	@echo "make clean       清理缓存与 output"

install:
	$(PIP) install -r requirements.txt

install-tf:
	$(PIP) install tensorflow tensorflow-datasets Pillow tensorboard

test:
	$(PY) -m pytest tests -q
	$(PY) run_tests.py

chapter00:
	$(PY) chapters/00_ml_basics.py
chapter01:
	$(PY) chapters/01_tensors_autograd.py
chapter02:
	$(PY) chapters/02_linear_regression.py
chapter03:
	$(PY) chapters/03_mlp_mnist.py
chapter04:
	$(PY) chapters/04_cnn_cifar10.py
chapter05:
	$(PY) chapters/05_text_imdb.py
chapter06:
	$(PY) chapters/06_transfer_learning.py
chapter07:
	$(PY) chapters/07_callbacks_tensorboard.py
chapter08:
	$(PY) chapters/08_save_and_export.py
chapter09:
	$(PY) chapters/09_capstone_image_classifier.py
chapter10:
	$(PY) chapters/10_edge_raspberry_pi.py

chapters:
	$(PY) run_all_chapters.py

notebooks:
	$(PY) teaching/build_notebooks.py

download:
	$(PY) scripts/download_data.py --dataset mnist cifar10 imdb tf_flowers

clean:
	@echo "Cleaning __pycache__, output, models..."
	-rm -rf output models/saved models/checkpoints
	-find . -type d -name __pycache__ -prune -exec rm -rf {} +
	-find . -name "*.pyc" -delete
