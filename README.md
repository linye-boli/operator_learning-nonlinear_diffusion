# 非线性扩散问题的算子学习

## 算子学习方法：

傅里叶神经算子（FNO）与深度算子网络（DON）作为微分算子学习的代表性方法，为解决复杂物理系统的跨条件泛化难题提供了新范式。FNO基于谱域全局卷积核，通过傅里叶变换捕捉多尺度场演化的长程依赖性；DON通过隐式基函数分解与系数预测实现高维函数空间的高效映射。现有算子学习方法在线性及弱非线性场景中已展现优势，但在多尺度、强非线性问题中仍面临挑战。

本项目提出两种Fourier-DON变体架构，将FNO与DON两者结合，以学习从方程条件到特定时间点辐射扩散方程解的映射：第一类使用FNO生成基函数，并采用全连接网络处理系数；第二类则采用逐元素特征组合后接FNO解码器。相比传统数值方法（如有限元法），Fourier-DON要更加快速、准确且便于推广，能够实现复杂物理系统中的高效模拟。

## 非线性辐射扩散问题：

非线性辐射扩散问题是一类典型的多尺度强耦合输运方程，其核心在于描述辐射能量与物质能量通过光子输运产生的非线性能量交换过程。该过程的控制方程可表述为：

### 单温问题：

$$
\begin{aligned}
   & \frac{\partial E}{\partial t}-\nabla\cdot(D_L\nabla E) = 0, \quad(x,y,t)\in\Omega\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = \beta(x,y,t), \quad(x,y,t)\in\lbrace x=0\rbrace\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = 0, \quad(x,y,t)\in\partial\Omega\setminus\lbrace x=0\rbrace\times[0,1] \\
   & E|_{t=0} = g(x,y,0)
\end{aligned}
$$

其中 $\Omega = [0,1]\times[0,1]$ ；辐射扩散系数 $D_L$ 选用限流形式，即 $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}$ 。

### 双温问题：

$$
\begin{aligned}
   & \frac{\partial E}{\partial t} - \nabla \cdot (D_L \nabla E) = \sigma_{\alpha}(T^4 - E), \quad(x,y,t)\in\Omega\times[0,1] \\
   & \frac{\partial T}{\partial t} - \nabla \cdot (K_L \nabla T) = \sigma_{\alpha}(E - T^4), \quad(x,y,t)\in\Omega\times[0,1] \\
   & 0.5E + D_L \nabla E \cdot n = \beta(x,y,t), \quad (x,y,t) \in \lbrace x=0 \rbrace \times [0,1] \\
   & 0.5E + D_L \nabla E \cdot n = 0, \quad (x,y,t) \in \partial\Omega \setminus \lbrace x=0 \rbrace \times [0,1] \\
   & K_L \nabla T \cdot n = 0, \quad (x,y,t) \in \partial\Omega \times [0,1] \\
   & E\vert_{t=0} = g(x,y,0) \\
   & T^4\vert_{t=0} = g(x,y,0)
\end{aligned}
$$

其中 $\Omega = [0,1]\times[0,1]$ ；辐射扩散系数 $D_L, K_L$ 同样选用限流形式，即 $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}, K_L = \frac{T^4}{T^{3/2}z+T^{5/2}|\nabla T|}$ 。

对于上述单温、双温问题，电离度函数 $Z$ 采用双方形，即在 $\Omega$ 内的两个0.25×0.25的方形区域中， $Z=9$ ；其他时候 $Z=1$ 。

初值条件采用常数初值，即 $g(x,y,t) = 0.01$ ；边值条件采用线性边值，即当 $t<t_1$ 时， $\beta(x,y,t)=\frac{\beta_{\text{max}}}{t_1} t$ ；当 $t\leq t_1$ 时， $\beta(x,y,t)=\beta_{\text{max}}$ 。

### 算子学习问题：

本项目需要研究的六个算子学习问题如下：

|                    | Tasks                          |
|--------------------|--------------------------------|
| single-temperature | ① $Z \rightarrow E$            |
|                    | ② $Z \times t_1 \rightarrow E$ |
|                    | ③ $Z \times t_1 \times \beta_{\text{max}} \rightarrow E$ |
| single-temperature | ④ $Z \rightarrow E, T$         |
|                    | ⑤ $Z \times t_1 \rightarrow E, T$ |
|                    | ⑥ $Z \times t_1 \times \beta_{\text{max}} \rightarrow E, T$ |

## Fourier-DON算法：

### 算法设计：

本项目的目标是找一个替代模型，用于处理多输入算子 $𝒢:𝒳_1\times 𝒳_2\times ... \times 𝒳_n\rightarrow 𝒴$ ，其中 $𝒳_1\times 𝒳_2\times ... \times 𝒳_n$ 表示 $n$ 个不同的输入函数空间，𝒴是输出函数空间。以上述算子学习问题②为例，假设有 $N$ 对参考数值解 $\{Z^{(k)},\beta^{(k)},E^{(k)}\} ,k=1,...,N$ ，则 $Z^{(k)} ∈𝒳_1,\beta^{(k)} ∈𝒳_2,E^{(k)} ∈𝒴$ ，目标是训练一个神经网络模型 $𝒢_\theta$ ，其中 $\theta$ 表示神经网络的可学习参数，通过最小化损失函数𝒞来近似𝒢：

$$
\begin{equation}
   \min_{\theta}\frac{1}{N}\sum_{k=1}^N 𝒞(𝒢_{\theta}(Z^{(k)},\beta^{(k)}),E^{(k)}).
\end{equation}
$$

普通的DON由两个网络组成：(1)主干网络，处理坐标作为输入；(2)分支网络，接收离散函数作为输入。为了适应本项目的需求，对分支网络进行修改，使其接受离散电离度函数 $Z$ ；同时修改主干网络，以处
理源函数 $\xi$ 的离散表示。

该算法的关键组成部分在于Fourier层，它结合了核积分变换和逐点变换，随后应用非线性激活函数 $\sigma$ ：

$$
\begin{equation}
   𝐕^{(l+1)} = \sigma(ℱ^{-1}(𝐑\cdot ℱ(𝐕^{(l)})) + 𝒲(𝐕^{(l)})).
\end{equation}
$$

其中，ℱ是快速傅里叶变换（FFT）， $ℱ^{-1}$ 是其逆变换， $𝐕^{(l)}∈ℝ^{m\times m\times c_{\text{in}}}$ 是输入， $𝐕^{(l+1)}∈ℝ^{m\times m\times c_{\text{out}}}$ 是输出， $𝐑∈ℝ^{m\times m\times c_{\text{in}}\times c_{\text{out}}}$ 是傅里叶空间中的可学习核。这里， $m$ 表示空间维度， $c_{\text{in}}$ 表示输入特征的数量， $c_{\text{out}}$ 表示输出特征的数量。逐点变换𝒲由FCN实现，𝐑的高阶模式被截断以减少计算成本。

下面使用单温任务来描述两个Fourier-DON变体。对于双温任务，只需使用两个Fourier-DON网络来学习目标函数 $𝐄,𝐓∈ℝ^{m\times m}$ 即可。

#### 第一类Fourier-DON：

设 $𝐙∈ℝ^{m\times m}$ 表示输入到分支网络 $B_\theta$ 的离散电离度函数 $Z$ ， $\xi∈ℝ^p$ 表示输入到主干网络 $T_\theta$ 的离散源函数 $\beta$ 。分支和主干网络的输出为：

$$
\begin{aligned}
   & 𝐕 = B_\theta (𝐙)∈ℝ^{m\times m\times c}, \\
   & 𝛚 = T_\theta (\xi)∈ℝ^c,
\end{aligned}
$$

其中 $c$ 是通道数，在非线性基函数逼近的背景下，每个通道对应一个基函数。分支网络 $B_\theta$ 由多个Fourier层组成，而主干网络 $T_\theta$ 是一个FCN。

离散化的目标函数 $𝐄∈ℝ^{m\times m}$ 近似为：

$$
\begin{equation}
   𝐄̃ = \sum_i 𝛚_i 𝐕_i,
\end{equation}
$$

其中 $𝐕_i∈ℝ^{m\times m}$ 是第 $i$ 个离散化的基函数， $𝛚_i∈ℝ$ 是其系数。

#### 第二类Fourier-DON：

分支网络和主干网络都将输入从物理空间映射到特征空间，然后通过逐元素乘法将它们的输出组合起来生成𝐕，作为解码器网络 $\Phi_\theta$ 的输入。解码器类似于FNO，由多个Fourier层和一个最终投影层组成，将𝐕转换为物理空间中的预测输出𝐄：

$$
\begin{aligned}
   & 𝐁 = B_\theta (𝐙)∈ℝ^{m\times m\times c}, \\
   & 𝐗 = T_\theta (\xi)∈ℝ^c, \\
   & 𝐕_{i,j} = 𝐁_{i,j}⊙𝐗, \\
   & 𝐄̃ = \Phi_\theta (𝐕^{(0)})
\end{aligned}
$$

其中 $i,j∈[0,m-1]$ 表示𝐁和𝐕的空间索引，⊙表示逐元素乘积。

### 数据集：

对于单温问题，取129×129的网格点，设置时间步长为0.001，皮卡迭代至收敛极限为0.001或迭代100步，将有限元法求出的结果作为参考解。对于双温问题，取257×257的网格点，设置时间步长为0.001，皮卡迭代至收敛极限为0.01或迭代100步，将有限元法求出的结果作为参考解。

输入还包括随机采样的 $Z,t_1,\beta_{\text{max}}$ 。参数 $Z$ 的两个方形区域左下角坐标从(0,1)中随机采样，参数 $t_1$ 在[0,1]中随机采样，参数 $\beta_{\text{max}}$ 在[9,11]中随机采样，用于构建源函数 $\beta(x,y,t)$ 。对于仅以 $Z$ 为输入的任务，设 $t_1=0.5,\beta_{\text{max}}=10$ ；对于仅以 $Z,t_1$ 为输入的任务，设 $\beta_{\text{max}}=10$ 。

### 模型训练：

#### 预处理：

两种架构的预处理过程相同。对于分支网络，输入𝐙被缩放到范围(0,1)，并与相应的二维网格坐标 $X,Y∈ℝ^{m\times m}$ 拼接，形成 $[Z,X,Y]∈ℝ^{m\times m\times 3}$ 。同样地，主干网络的输入 $t_1∈ℝ$ 和 $\beta_{\text{max}}∈ℝ$ 也被缩放到(0,1)，并拼接成 $[t_1,\beta_{\text{max}}]∈ℝ^2$ 。

#### 模型参数：

##### 第一类Fourier-DON：

分支网络由一个线性层组成，该层将 $ℝ^{m\times m\times 3}$ 映射到 $ℝ^{m\times m\times 32}$ 。随后是四个Fourier层，每层包含12个模式和32个通道，层内的逐点变换块𝒲实现为一个两层FCN，每层有32个隐藏单元。

主干网络是一个四层FCN，每层有32个隐藏单元。

GeLU激活函数应用于除最后一层外两个网络的所有层。

##### 第二类Fourier-DON：

分支线性和主干层均通过一个线性层实现，该层将 $ℝ^{m\times m\times 3}$ 映射到 $ℝ^{m\times m\times 32}$ ，然后进行逐元素乘法。输出 $𝐕∈ℝ^{m\times m\times 32}$ 被送入一个FNO解码器，该解码器包含四个Fourier层（每个层与上述第一类中的相同）以及一个包含32个隐藏单元的两层FCN。

对于具有固定源函数 $\beta(x,y,t)$ 的任务，主干网络被省略，分支网络在傅里叶层之后增加了一个额外的线性投影层，以直接将输出映射到解空间。

#### 损失函数：

损失函数定义为相对L2范数误差：

$$
\begin{equation}
   ‖L‖₂ = \frac{1}{N} \sum_{k=1}^N \frac{‖𝐄^{(k)}-𝐄̃^{(k)}‖₂}{‖𝐄^{(k)}‖₂},
\end{equation}
$$

其中 $N$ 表示样本数， $𝐄^{(k)}$ 是第 $k$ 个FEM参考解， $𝐄̃^{(k)}$ 是神经网络的相应预测。

#### 训练参数：

两种架构的模型参数均使用小批量梯度下降的Adam优化器。采用余弦退火调度器，设置初始学习率为0.001，训练过程在100个周期内完成。默认情况下所有任务使用4个样本的批次大小，600个训练样本和100个测试样本。

为了确保硬件和软件的一致性，所有训练和预测过程均在配备有n Intel Xeon W-2255 10核处理器和NVIDIA RTX A6000 GPU的PC上执行。神经网络使用Pytorch实现，并采用随机种子以保证实验的可重复性和再现性。
   









# Operator Learning for Nonlinear Diffusion Problems

This repository contains scripts to reproduce the results from the paper on operator learning for solving nonlinear diffusion problems. Follow the instructions below to set up the project, run experiments, and process results.

## Project Structure

```
operator_learning-nonlinear_diffusion/
├── dataset/
│   ├── nd/
│   └── nd_seq/
├── result/
│   ├── exps/
│   ├── seq_exps/
│   ├── figs/
│   └── result_process.py
├── src/
│   ├── train.py
│   ├── nets.py
│   ├── utils.sh
│   ├── default_exps.sh
│   ├── nlayer_exps.sh
│   ├── ntrain_exps.sh
│   ├── modes_exps.sh
│   ├── width_exps.sh
│   ├── superres_exps.sh
│   └── seq_exps.sh
├── requirements.txt
└── README.md
```

## Prerequisites

- A system with a compatible GPU (ensure valid GPU device IDs are available).
- Python (version compatible with dependencies, e.g., Python 3.8+) and Bash installed.
- Required Python dependencies listed in `requirements.txt` (e.g., `torch`, `numpy`, `scipy`, `matplotlib`).
- Ensure Pytorch installed with GPU support.
- Access to the dataset and results (download link below).

## Setup

1. **Download Dataset and Results**:
   - Access the dataset and results at: [https://pan.baidu.com/s/1CEs6UBiWCt3dzjk-vs98og?pwd=nrde](https://pan.baidu.com/s/1CEs6UBiWCt3dzjk-vs98og?pwd=nrde).
   - Unzip `dataset.zip` and `result.zip`.
   - Place the extracted `dataset/` and `result/` folders in the root directory as shown in the project structure.

2. **Verify Project Structure**:
   - Ensure the project directory matches the structure above, including the `requirements.txt` file.

3. **Install Dependencies**:
   - Create a virtual environment (recommended to avoid conflicts):
     ```bash
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     ```
   - Install dependencies from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

## Running Experiments

The following scripts generate results for specific tables and figures in the paper:

| Script                | Generates Results For                     |
|-----------------------|-------------------------------------------|
| `default_exps.sh`     | Table II, Table III, Table IV, Fig. 7, Fig. 8 |
| `nlayer_exps.sh`      | Fig. 9                                   |
| `ntrain_exps.sh`      | Fig. 9                                   |
| `modes_exps.sh`       | Fig. 9                                   |
| `width_exps.sh`       | Fig. 9                                   |
| `superres_exps.sh`    | Table V                                  |
| `seq_exps.sh`         | Fig. 10, Table VI                        |

### Steps to Run Experiments

1. Navigate to the `src/` directory:
   ```bash
   cd src
   ```

2. Execute the desired script, specifying the GPU device ID (e.g., `0`, `1`):
   ```bash
   bash <script_name>.sh device=<ID>
   ```
   Example:
   ```bash
   bash default_exps.sh device=0
   ```

3. Repeat for each script as needed (`default_exps.sh`, `nlayer_exps.sh`, etc.).

### Using the `train.py` Script

The `train.py` script trains and performs inference with Fourier Neural Operator models (`FNO2d`, `FDON2d`, `FDON2d_II`) for tasks like heat diffusion. 

**Key Features**:
- **Training**: Trains models using L2 loss, Adam optimizer, and cosine annealing learning rate scheduling. Supports `FNO2d` (input: initial conditions) and `FDON2d`/`FDON2d_II` (inputs: initial conditions and boundary conditions).
- **Inference**: Computes test predictions, relative L2 loss, and inference times (GPU/CPU).
- **Output**: Saves model weights, predictions, loss dynamics, and inference times to `../result/<task>/<component>/`.

**Command Example**:
```bash
python train.py --task heat-1T-zsquares --arch fno --num-train 600 --num-test 100 --batch-size 4 --device 0
```

**Key Arguments**:
- `--task`: Task name (e.g., `heat-1T-zsquares`, `heat-1T-zsquares-t1`).
- `--arch`: Model architecture (`fno`, `fdon1`, `fdon2`).
- `--num-train`/`--num-test`: Number of training/test samples.
- `--batch-size`: Batch size for training.
- `--device`: GPU device ID.
- `--epochs`: Number of training epochs (default: 100).
- `--lr`: Learning rate (default: 1e-3).
- `--modes`/`--width`: Fourier modes and network channels.
- Full list available via `python train.py --help`.

## Processing Results

After completing all experiments, process the results to generate the final tables and figures:

1. Navigate to the `result/` directory:
   ```bash
   cd result
   ```

2. Run the result processing script:
   ```bash
   python result_process.py
   ```

This script aggregates experiment outputs and produces the results corresponding to the paper’s tables and figures.

## Troubleshooting

- **GPU Errors**: Verify the GPU device ID and ensure CUDA drivers are compatible with the `torch` version in `requirements.txt`.
- **Missing Dependencies**: If errors occur, ensure all packages in `requirements.txt` are installed. Check the paper for additional requirements.
- **Incomplete Results**: Ensure all experiments have run successfully before processing results.
- **File Structure Issues**: Confirm `dataset/`, `result/`, and `requirements.txt` are correctly placed.
- **Task Errors**: Use supported tasks (e.g., `heat-1T-zsquares`) as listed in `train.py`.

## Additional Notes

- The `result_process.py` script assumes all experiments have completed successfully.
- Refer to the paper for detailed experiment descriptions and expected outputs.
- For large-scale experiments, monitor system resources to prevent crashes.
