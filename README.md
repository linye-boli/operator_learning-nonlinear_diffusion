# 非线性扩散问题的算子学习

## 算子学习方法：

傅里叶神经算子（FNO）与深度算子网络（DON）作为微分算子学习的代表性方法，为解决复杂物理系统的跨条件泛化难题提供了新范式。FNO基于谱域全局卷积核，通过傅里叶变换捕捉多尺度场演化的长程依赖性；DON通过隐式基函数分解与系数预测实现高维函数空间的高效映射。现有算子学习方法在线性及弱非线性场景中已展现优势，但在多尺度、强非线性问题中仍面临挑战。

本项目提出两种Fourier-DON架构，将FNO与DON两者结合，以学习从方程条件到特定时间点辐射扩散方程解的映射：第一类使用FNO生成基函数，并采用全连接网络处理系数；第二类则采用逐元素特征组合后接FNO解码器。相比传统数值方法（如有限元法），Fourier-DON要更加快速、准确且便于推广，能够实现复杂物理系统中的高效模拟。

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

对于上述单温、双温问题，电离度函数 $Z$ 采用双方形，即当 $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ 或 $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$ 时， $Z=10$ ；其他时候 $Z=1$ 。

初边值条件采用常数初值+线性边值，即 $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$ 。

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

## Fourier-DON算法设计：

本项目的目标是找一个替代模型，用于处理多输入算子 $G: X_1\times X_2\times ... \times X_n\rightarrow Y$ ，其中 $X_1\times X_2\times ... \times X_n$ 表示 $n$ 个不同的输入函数空间， $Y$ 是输出函数空间。以上述算子学习问题②为例，假设有 $N$ 对参考数值解 $\{Z^{(k)},\beta^{(k)},E^{(k)}\}_{k=1}^N$ ，则 $Z^{(k)}\in X_1$ ， $\beta^{(k)}\in X_2$ 对应， $E^{(k)}\in Y$ 对应，目标是训练一个神经网络模型 $G_{\theta}$ ，其中 $\theta$ 表示神经网络的可学习参数，通过最小化损失函数 $C$ 来近似 $G$ ：

$$
\begin{equation}
   \min_{\theta}\frac{1}{N}\sum_{k=1}^N C(G_{\theta}(Z^{(k)},\beta^{(k)}),E^{(k)}).
\end{equation}
$$

该算法的关键组成部分在于Fourier层，它结合了核积分变换和逐点变换，随后应用非线性激活函数 $\sigma$ ：
\begin{equation}
   V^{(l+1)} = \sigma(F^{-1}(R\cdot F(V^{(l)})) + W(V^{(l)})).
\end{equation}









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
