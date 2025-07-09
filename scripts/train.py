#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 实现DINOv2特征提取

# In[2]:


import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# 配置类：设置路径、模型名称、图像大小和设备等参数
class CFG:
    DATA_DIR = "/kaggle/input/planttraits2024"  # 数据集路径
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")  # 训练图像路径
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")  # 测试图像路径
    OUTPUT_DIR = "/kaggle/working/"  # 嵌入保存路径

    DINO_MODEL_NAME = "facebook/dinov2-base"  # 使用的DINOv2模型
    IMAGE_SIZE = 224  # 图像尺寸
    BATCH_SIZE = 64  # 批处理大小，依赖GPU内存
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备选择

# 加载CSV文件中的元数据
train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "test.csv"))

# 加载DINOv2图像处理器和模型
print(f"Loading DINOv2 model: {CFG.DINO_MODEL_NAME} on device: {CFG.DEVICE}")
processor = AutoImageProcessor.from_pretrained(CFG.DINO_MODEL_NAME)
model = AutoModel.from_pretrained(CFG.DINO_MODEL_NAME).to(CFG.DEVICE).eval()

# 构造图像路径的函数
def get_image_path(image_id, is_test=False):
    base_dir = CFG.TEST_IMAGES_DIR if is_test else CFG.TRAIN_IMAGES_DIR
    return os.path.join(base_dir, f"{image_id}.jpeg")

# 提取图像嵌入的函数
def extract_embeddings(df, is_test=False):
    image_paths = df['id'].apply(lambda x: get_image_path(x, is_test)).tolist()
    all_embeddings = []

    # 按批处理图像以节省内存
    for i in tqdm(range(0, len(image_paths), CFG.BATCH_SIZE), desc=f"Extracting {'Test' if is_test else 'Train'} Embeddings"):
        batch_paths = image_paths[i : i + CFG.BATCH_SIZE]

        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(CFG.DEVICE)
        except Exception as e:
            print(f"Error loading batch starting at index {i}: {e}")
            batch_size = len(batch_paths)
            embedding_dim = model.config.hidden_size
            placeholder_embeddings = np.zeros((batch_size, embedding_dim))
            all_embeddings.append(placeholder_embeddings)
            continue

        # 执行前向推理
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 提取CLS token输出
            all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)

# 提取训练集嵌入并保存
print("Starting Train Embedding Extraction...")
train_embeddings = extract_embeddings(train_df, is_test=False)
train_embedding_path = os.path.join(CFG.OUTPUT_DIR, "train_dino_embeddings.npy")
np.save(train_embedding_path, train_embeddings)
print(f"Train embeddings saved to {train_embedding_path}, shape: {train_embeddings.shape}")

# 提取测试集嵌入并保存
print("\nStarting Test Embedding Extraction...")
test_embeddings = extract_embeddings(test_df, is_test=True)
test_embedding_path = os.path.join(CFG.OUTPUT_DIR, "test_dino_embeddings.npy")
np.save(test_embedding_path, test_embeddings)
print(f"Test embeddings saved to {test_embedding_path}, shape: {test_embeddings.shape}")


# 数据预处理

# In[3]:


from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import torch

# 配置类：定义路径和设备
class CFG:
    DATA_DIR = "/kaggle/input/planttraits2024"  # 原始数据路径
    OUTPUT_DIR = "/kaggle/working/"  # 输出路径
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 是否使用GPU

# 加载训练和测试数据
print("Loading data from CSV files...")
train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "test.csv"))
print("Data loaded successfully.")

# 定义表格特征处理配置
class TabularCFG:
    TARGET_BASE_COLS = ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']  # 目标列
    SKEWED_TARGETS = ['X11', 'X18', 'X26', 'X50', 'X3112']  # 偏态目标列

    OUTLIER_CLIP_LOWER_QUANTILE = 0.005  # 异常值下界分位数
    OUTLIER_CLIP_UPPER_QUANTILE = 0.995  # 异常值上界分位数
    POLYNOMIAL_DEGREE = 2  # 多项式扩展次数

# 设置表格特征列（排除 id）
TabularCFG.TABULAR_FEATURES = [
    col for col in test_df.columns if col != 'id'
]

# 开始表格特征处理流水线
print("\nStarting data processing pipeline...")

# 对偏态目标变量做 log1p 变换
print("Applying log1p transformation to skewed target variables...")
for trait in TabularCFG.SKEWED_TARGETS:
    train_df[f'{trait}_log'] = np.log1p(train_df[f'{trait}_mean'])

# 对特征值做 clip，处理异常值
print(f"Clipping features...")
for col in tqdm(TabularCFG.TABULAR_FEATURES, desc="Calculating Clip Bounds"):
    lower = train_df[col].quantile(TabularCFG.OUTLIER_CLIP_LOWER_QUANTILE)
    upper = train_df[col].quantile(TabularCFG.OUTLIER_CLIP_UPPER_QUANTILE)
    train_df[col] = train_df[col].clip(lower, upper)
    test_df[col] = test_df[col].clip(lower, upper)

# 多项式特征扩展（包括二次项和交叉项）
print(f"Generating polynomial features...")
poly = PolynomialFeatures(degree=TabularCFG.POLYNOMIAL_DEGREE, include_bias=False, interaction_only=False)
train_poly_features = poly.fit_transform(train_df[TabularCFG.TABULAR_FEATURES])
test_poly_features = poly.transform(test_df[TabularCFG.TABULAR_FEATURES])

# 将生成的多项式特征转换为DataFrame
poly_feature_names = poly.get_feature_names_out(TabularCFG.TABULAR_FEATURES)
train_poly_df = pd.DataFrame(train_poly_features, columns=poly_feature_names, index=train_df.index)
test_poly_df = pd.DataFrame(test_poly_features, columns=poly_feature_names, index=test_df.index)

# 标准化处理（Z-score）
print("Applying Z-score standardization...")
scaler = StandardScaler()
train_tabular_scaled = scaler.fit_transform(train_poly_df)
test_tabular_scaled = scaler.transform(test_poly_df)

# 保存处理后的表格数据为 .npy 文件
print("\nSaving processed tabular data...")
train_tabular_path = os.path.join(CFG.OUTPUT_DIR, "train_tabular_processed.npy")
test_tabular_path = os.path.join(CFG.OUTPUT_DIR, "test_tabular_processed.npy")
np.save(train_tabular_path, train_tabular_scaled)
np.save(test_tabular_path, test_tabular_scaled)

print(f"Processed tabular data saved.")
print(f"Train shape: {train_tabular_scaled.shape}")


# 模型训练、推断和提交

# In[4]:


import gc
import catboost
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import os
import torch

# 配置类：用于存储常量路径和设备信息
class CFG:
    DATA_DIR = "/kaggle/input/planttraits2024"  # 数据路径
    OUTPUT_DIR = "/kaggle/working/"  # 输出路径
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 判断是否使用GPU

# 配置类：存储目标变量和偏斜变量名称
class TabularCFG:
    TARGET_COLS = ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']
    SKEWED_TARGETS = ['X11', 'X18', 'X26', 'X50', 'X3112']  # 偏态变量

# 加载预处理后的数据
print("Loading pre-processed features for Stage 2 modeling...")
train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "test.csv"))

# 对偏态变量进行log1p变换
for trait in TabularCFG.SKEWED_TARGETS:
    train_df[f'{trait}_log'] = np.log1p(train_df[f'{trait}_mean'])

# 加载图像嵌入和表格特征
train_embeddings = np.load(os.path.join(CFG.OUTPUT_DIR, "train_dino_embeddings.npy"))
test_embeddings = np.load(os.path.join(CFG.OUTPUT_DIR, "test_dino_embeddings.npy"))
train_tabular_scaled = np.load(os.path.join(CFG.OUTPUT_DIR, "train_tabular_processed.npy"))
test_tabular_scaled = np.load(os.path.join(CFG.OUTPUT_DIR, "test_tabular_processed.npy"))

# 数据融合：将图像嵌入和表格特征组合为一个DataFrame
print("Fusing image embeddings and tabular features into a DataFrame...")
num_tabular_features = train_tabular_scaled.shape[1]
tabular_cols = [f'tab_{i}' for i in range(num_tabular_features)]
X = pd.DataFrame(train_tabular_scaled, columns=tabular_cols)
X_test = pd.DataFrame(test_tabular_scaled, columns=tabular_cols)
X['dino_embedding'] = list(train_embeddings)
X_test['dino_embedding'] = list(test_embeddings)

print(f"Final training feature matrix shape: {X.shape}")
print(f"Final test feature matrix shape: {X_test.shape}")
print("Feature format after fusion:")
print(X.head())

# 清理不再需要的变量释放内存
del train_embeddings, test_embeddings, train_tabular_scaled, test_tabular_scaled
gc.collect()

# 模型训练配置
class ModelCFG:
    N_SPLITS = 5
    SEED = 42
    CATBOOST_PARAMS = {
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 50,
        'embedding_features': ['dino_embedding'],  # 指定嵌入列名
        'task_type': 'GPU' if CFG.DEVICE == 'cuda' else 'CPU'
    }

kf = KFold(n_splits=ModelCFG.N_SPLITS, shuffle=True, random_state=ModelCFG.SEED)
oof_preds = {}
test_preds = {}
oof_scores = {}

# 初始化预测字典
for trait in TabularCFG.TARGET_COLS:
    oof_preds[trait] = np.zeros(len(train_df))
    test_preds[trait] = np.zeros(len(test_df))

# 主训练循环
plot_learning_curve_done = False 
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== FOLD {fold+1} / {ModelCFG.N_SPLITS} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

    for trait in TabularCFG.TARGET_COLS:
        print(f"--- Training model for trait: {trait} ---")
        is_log_target = trait in TabularCFG.SKEWED_TARGETS
        target_col = f'{trait}_log' if is_log_target else f'{trait}_mean'

        y = train_df[target_col] 
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = catboost.CatBoostRegressor(**ModelCFG.CATBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            use_best_model=True
        )

        # 仅第一次绘制学习曲线
        if not plot_learning_curve_done:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("darkgrid")
            evals_result = model.get_evals_result()
            train_r2 = evals_result['learn']['R2']
            val_r2 = evals_result['validation']['R2']
            epochs = range(1, len(train_r2) + 1)
            plt.figure(figsize=(12, 7))
            plt.plot(epochs, train_r2, 'b-', label='Train R2')
            plt.plot(epochs, val_r2, 'o-', label='Validation R2')
            plt.title(f'Learning Curve for Trait "{trait}" (Fold {fold+1})')
            plt.xlabel('Iterations')
            plt.ylabel('R2 Score')
            plt.legend()
            plt.show()
            plot_learning_curve_done = True

# 第二轮训练+预测（逻辑与前相同）
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== FOLD {fold+1} / {ModelCFG.N_SPLITS} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx] 

    for trait in TabularCFG.TARGET_COLS:
        print(f"--- Training model for trait: {trait} ---")
        is_log_target = trait in TabularCFG.SKEWED_TARGETS
        target_col = f'{trait}_log' if is_log_target else f'{trait}_mean'

        y = train_df[target_col] 
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = catboost.CatBoostRegressor(**ModelCFG.CATBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )

        val_preds = model.predict(X_val)
        fold_test_preds = model.predict(X_test)

        oof_preds[trait][val_idx] = val_preds
        test_preds[trait] += fold_test_preds / ModelCFG.N_SPLITS

    del X_train, X_val
    gc.collect()

# 输出交叉验证R2得分
print("\n===== OOF R2 SCORES =====")
for trait in TabularCFG.TARGET_COLS:
    is_log_target = trait in TabularCFG.SKEWED_TARGETS
    true_values = train_df[f'{trait}_mean'].values
    pred_values = np.expm1(oof_preds[trait]) if is_log_target else oof_preds[trait]
    score = r2_score(true_values, pred_values)
    oof_scores[trait] = score
    print(f"Trait {trait}: {score:.4f}")

avg_oof_score = np.mean(list(oof_scores.values()))
print(f"\nAverage OOF R2 Score: {avg_oof_score:.4f}")

# 创建提交文件
print("\nCreating submission file...")
submission_df = pd.DataFrame({'id': test_df['id']})
for trait in TabularCFG.TARGET_COLS:
    is_log_target = trait in TabularCFG.SKEWED_TARGETS
    submission_df[trait] = np.expm1(test_preds[trait]) if is_log_target else test_preds[trait]

# 防止负值
for trait in TabularCFG.TARGET_COLS:
    submission_df[trait] = submission_df[trait].clip(lower=0)

submission_path = os.path.join(CFG.OUTPUT_DIR, 'submission.csv')
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved to {submission_path}")
print(submission_df.head())

