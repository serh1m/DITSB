# DITSB-v2 Google Colab 大语言模型训练方案

本方案详细说明了如何将现有的 DITSB 架构部署到 Google Colab 环境中，完成从代码上传、环境配置、数据下载到模型训练的完整流程。

---

## 第一阶段：Colab 实例准备与存储挂载

### 1. 创建 Colab Notebook 并开启 GPU
1. 访问 [Google Colab](https://colab.research.google.com/) 并新建一个 Notebook。
2. 点击上方菜单栏 **"代码执行程序" (Runtime)** -> **"更改代码执行程序类型" (Change runtime type)**。
3. 硬件加速器选择 **GPU**。
   > **提示**：如果是免费用户，通常分配到 T4 CPU；如果拥有 Colab Pro/+，强烈建议选择 **A100 或 V100 GPU** 以及高内存 (High-RAM) 模式，这对于 LLM 训练和 O(1) 显存算法（DITSB-v2）加速最为明显。

### 2. 挂载 Google Drive
为了防止 Colab 实例掉线导致训练检查点（Checkpoints）丢失，**必须**将 Google Drive 挂载到环境，用于持久化保存模型权重。在 Notebook 的第一个 Cell 运行：
```python
from google.colab import drive
drive.mount('/content/drive')
```
*运行后会弹窗请求授权，允许即可。*

---

## 第二阶段：上传并同步代码架构

有以下两种推荐的方式将您的 `DITSB` 架构上传到 Colab：

### 方法 A：通过 Github 克隆（推荐 - 方便代码同步）
如果您已将桌面上的 `DITSB` 推送至个人的 Github（如 `serh1m/DITSB`）：
```bash
!git clone https://github.com/您的用户名/DITSB.git /content/DITSB
%cd /content/DITSB
```

### 方法 B：通过 Google Drive 上传压缩包（适合私有离线代码）
1. 在本地电脑，将桌面 `c:\Users\Administrator\Desktop\DITSB` 文件夹压缩为 `DITSB.zip`。
2. 将 `DITSB.zip` 上传至您的 Google Drive 根目录。
3. 在 Colab 中解压到本地运行目录 `/content` (比直接在网盘读取速度快几十倍)：
```bash
!unzip /content/drive/MyDrive/DITSB.zip -d /content/
%cd /content/DITSB
```

---

## 第三阶段：环境安装与配置

PyTorch 在 Colab 中默认已被安装。还需要安装依赖库和数据处理所需的库：
```bash
# 假设您在 DITSB 目录下
!pip install -r requirements.txt
!pip install datasets transformers accelerate pyyaml
```

---

## 第四阶段：下载训练数据与预处理

**关键点**：由于 Google Drive 的 I/O 读写速度受限，**千万不要**将海量训练数据下载到网盘内。应当直接下载到实例所带的高速 SSD 目录下（即 `/content/` 路径下）。

在 Colab 运行您写好的数据准备脚本：
```bash
# 执行数据准备脚本，并将输出路径指定为 Colab 本地高速目录
!python llm_training/prepare_data.py --output_dir=/content/dataset --max_samples=100000 
```

---

## 第五阶段：配置并启动训练

### 1. 修改配置文件 (`config_7b.yaml`)
如果需要，使用 Colab 左侧的文件浏览器打开 `llm_training/config_7b.yaml`，确保以下核心路径设定符合 Colab 环境：
- **数据路径**: 指向刚才准备的大数据集 `/content/dataset`。
- **权重保存路径**: 指向 Google Drive，防止丢失 `/content/drive/MyDrive/DITSB_checkpoints`。

您也可以直接通过脚本修改或在 Notebook 覆盖参数：
```python
import yaml

config_path = 'llm_training/config_7b.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 修改关键路径
config['training']['data_dir'] = '/content/dataset'
config['training']['checkpoint_dir'] = '/content/drive/MyDrive/DITSB_checkpoints'
config['training']['batch_size'] = 8 # 根据 Colab 提供显卡类型微调

with open(config_path, 'w') as f:
    yaml.dump(config, f)
```

### 2. 预估性能（可选项）
运行性能预估脚本验证在所分发 GPU 上的可行性：
```bash
!python llm_training/predict_performance.py --config llm_training/config_7b.yaml
```

### 3. 一键启动大模型训练
开始训练，并将所有日志实时打印到 Notebook 单元格中：
```bash
!python llm_training/train_llm.py --config llm_training/config_7b.yaml
```

---

## 第六阶段：防掉线与断点续训策略

### 1. Colab 网页防休眠 (Anti-Disconnect)
为了防止挂机时 Colab 断开连接，可以在浏览器的 **F12 开发者控制台 (Console)** 中输入以下 JavaScript 代码回车，定期模拟点击操作：
```javascript
function ConnectButton(){
    console.log("正在尝试点击防止掉线...");
    document.querySelector("colab-connect-button").shadowRoot.getElementById("connect").click()
}
setInterval(ConnectButton, 60000); // 每1分钟点击一次
```

### 2. 断点续训 (Resume Training)
如果由于 Colab 超时 (最高 12-24 小时) 被强制中断。在下次重启环境时，只需重新完成**第一到第四阶段**环境准备后，在训练命令中加上复原参数（如果有在 `train_llm.py` 内实现）：
```bash
!python llm_training/train_llm.py --config llm_training/config_7b.yaml --resume_from_checkpoint /content/drive/MyDrive/DITSB_checkpoints/latest
```
这样您的进度就会从 Google Drive 中恢复并继续了。
