# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行主训练脚本（Part 1）
python train.py

# 3. 运行BN对比实验（Part 2）
python train_bn_comparison.py

# 4. 生成Loss Landscape图
python loss_landscape.py

# 5. 可视化卷积核
python visualize_filters.py