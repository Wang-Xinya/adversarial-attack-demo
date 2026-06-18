# Adversarial Attack Demo (MNIST)

交互式对抗攻击（FGSM / PGD / MI-FGSM）与启发自随机平滑的经验性防御的 MNIST 演示系统
基于 PyTorch 和 PyQt5 构建，提供单张攻击和批量评估两种模式，直观展示对抗样本生成与防御效果

## 1. 功能特点

- 三种白盒攻击：FGSM、PGD、MI-FGSM
- 一种经验性防御：基于噪声投票（启发自随机平滑）
- 图形界面（PyQt5）：
  - **单张攻击**：随机加载测试图像，调节参数，显示原始图像、扰动放大图、对抗样本及预测置信度
  - **批量评估**：自动遍历 7 个扰动半径（ε），对比防御前后鲁棒准确率曲线及防御增益

## 2. 依赖环境

- Python 3.8+
- PyTorch
- torchvision
- PyQt5
- matplotlib
- numpy

安装命令：
```bash
pip install torch torchvision matplotlib PyQt5 numpy
```

## 3. 运行步骤

1. 克隆仓库：
```bash
git clone https://github.com/your-username/adversarial-attack-demo.git
cd adversarial-attack-demo
```

2. 启动程序：
```bash
python -m ui.main_window
```

首次运行会自动下载 MNIST 数据（约 60MB）并训练一个 LeNet 模型，之后即可使用界面

## 4. 项目结构

```
.
├── attacks/         # FGSM, PGD, MI-FGSM 扰动函数
├── data/            # MNIST 数据加载
├── defenses/        # 随机平滑预测与批量评估
├── models/          # LeNet 网络定义
├── utils/           # 训练、评估、绘图等通用工具
├── ui/              # PyQt5 主界面
├── demo_attack.py          # 命令行攻击演示
├── demo_defense.py  # 命令行防御演示
└── README.md
```
