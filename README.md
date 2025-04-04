# LLM4AD: Logical Anomaly Detection using Large Language Models

![概要図](assets/teaser.png)

This repository contains the implementation of a novel approach for logical anomaly detection in images using Large Language Models (LLMs).

## Overview

Logical anomaly detection is a challenging task in computer vision where the goal is to detect anomalies based on logical rules (e.g., correct number and placement of objects) rather than structural defects. Our approach leverages the powerful coding capabilities of LLMs to automatically generate computer vision programs that can detect logical anomalies from natural language descriptions of normal conditions.

### Key Features

- Natural language-based specification of normal conditions
- Automatic program generation using LLMs
- Integration with state-of-the-art computer vision components
- Evaluation on the MVTec LOCO AD dataset

## Background

Traditional anomaly detection approaches often rely on learning from a large number of normal samples. However, for logical anomalies, it is often more practical and intuitive to specify normal conditions in natural language rather than collecting numerous normal samples. Our approach bridges this gap by using LLMs to convert natural language descriptions into executable computer vision programs.

## Results
![実験結果](assets/result.png)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv llm4ad_env
source llm4ad_env/bin/activate  # On Linux/Mac
# or
.\llm4ad_env\Scripts\activate  # On Windows
```

2. Install PyTorch:
```bash
# For CPU
pip install torch torchvision torchaudio

# For GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Select the created virtual environment as the kernel.

3. Open the demo notebooks in the `notebooks/` directory to run the examples.

## Demo

The demo notebooks demonstrate how to:
- Define normal conditions using natural language
- Generate and execute anomaly detection programs
- Evaluate results on sample images

## Requirements

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU support)
- Other dependencies listed in `requirements.txt`

## License

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Reference

[1] User Name, '大規模言語モデルを用いたプログラム自動生成による論理的異常の画像検知'

## Author

[takah1r0jp](https://github.com/takah1r0jp)
