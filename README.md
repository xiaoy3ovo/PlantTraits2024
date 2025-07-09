# PlantTraits2024: Trait Classification using DINOv2 + CatBoost

## 📘 项目简介
本项目用于处理 [Kaggle PlantTraits2024](https://www.kaggle.com/competitions/planttraits2024) 数据，利用 DINOv2 提取图像特征，并结合 CatBoost 实现植物性状分类任务。

## 🧩 项目结构
- `notebook/main.ipynb`: 包含数据处理、特征提取与建模的完整 Notebook。
- `scripts/train.py`: 从 Notebook 中导出的训练脚本。
- `results/`: 存储最终预测结果或评估指标。

## 🛠️ 安装依赖
```bash
pip install -r requirements.txt
```

## 🚀 运行方式
```bash
# 运行 Notebook
jupyter notebook notebook/main.ipynb

# 或运行脚本
python scripts/train.py
```

## 📊 示例结果
示例输出图像保存在 `images/` 文件夹中。

## 📄 License
MIT License
