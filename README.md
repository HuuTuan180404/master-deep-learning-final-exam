# Pose Estimation for MSLR CSLR Track

This repository is the official implementation of the following paper.

Paper Title: **[A Signer-Invariant Conformer and Multi-Scale Fusion Transformer for Continuous Sign Language Recognition](https://arxiv.org/abs/2508.09372)**
Original repo: **[Original code](https://github.com/rezwanh001/MSLR-Pose86K-CSLR-Isharah)**

## Update_MSLR-2025 (follow this folder for updated code)

Follow these steps to set up the environment and get started:

1. **Clone the repository**:

   ```bash
   git clone --single-branch -b master https://github.com/HuuTuan180404/master-deep-learning-final-exam.git
   ```

   ```bash
   cd Update_MSLR-2025
   ```

2. **Download the dataset** from [here](https://drive.google.com/file/d/1RicLOS1JqO8FeY7BPPCfGF9-zdupbjTZ/view?usp=sharing).
   - This dataset is a 10% subset of the original experimental dataset used in the project.
   - After downloading, extract the archive and place the extracted dataset in the `Update_MSLR-2025/data` folder.

3. **Set up the Python environment**:
   - Install `virtualenv`:

     ```bash
     pip install virtualenv
     ```

   - Create a virtual environment and activate it:

     ```bash
     python<version> -m venv pose
     source pose/bin/activate  # On Windows: pose\Scripts activate
     ```

   - Install the required dependencies:
     ```bash
     pip install torch==1.13 torchvision==0.14 tqdm numpy==1.23.5 pandas opencv-python
     git clone --recursive https://github.com/parlance/ctcdecode.git
     cd ctcdecode && pip install .
     ```

4. **Run the code for both training and inference**
   - mode = SI (train):

     ```bash
     python run.py --train --mode SI --model SOTA_CSLR
     ```

   - mode = SI (infer)

     ```bash
     python run.py --infer --mode SI --model SOTA_CSLR
     ```

     ***

   - mode = US (train)

     ```bash
     python run.py --train --mode US --model AdvancedSignLanguageRecognizer
     ```

   - mode = US (infer)

     ```bash
     python run.py --infer --mode US --model AdvancedSignLanguageRecognizer
     ```

---

### Model Performance (US Mode)

    | Model                              | Mode   | Dev (WER)   | Test (WER)  |
    | ---------------------------------- | ------ | ----------- | ----------- |
    | llm_advslowfast                    | US     | 93.0663     | ...         |
    | gcn_transformer                    | US     | 91.7951     | ...         |
    | mixllama                           | US     | 86.9029     | ...         |
    | LLM Backbone (DistilBERT)          | US     | 81.7026     | ...         |
    | slowfast                           | US     | 81.3174     | ...         |
    | LSTM                               | US     | 79.9307     | ...         |
    | SignLanguageConformer              | US     | 77.5039     | ...         |
    | SignLanguageRecognizer             | US     | 74.9614     | ...         |
    | SOTA_CSLR                          | US     | 64.4838     | ...         |
    | MambaSignLanguageRecognizer        | US     | 59.514      | ...         |
    | **AdvancedSignLanguageRecognizer** | **US** | **55.0847** | **47.7756** |


### Model Performance (SI Mode)

    | Model                          | Mode   | Dev (WER)  | Test (WER)  |
    | ------------------------------ | ------ | ---------- | ----------- |
    | llm_advslowfast                | SI     | 43.8955    | 72.2365     |
    | MambaSignLanguageRecognizer    | SI     | 29.3149    | 37.2774     |
    | AdvancedSignLanguageRecognizer | SI     | 27.5362    | 33.9069     |
    | mixllama + slowfastllm         | SI     | 30.1274    | 46.9831     |
    | mixllama                       | SI     | 21.8270    | 51.2139     |
    | LSTM                           | SI     | 17.0180    | 26.0755     |
    | slowfastllm                    | SI     | 16.7106    | 42.5878     |
    | SignLanguageConformer          | SI     | 16.2495    | 26.6290     |
    | SignLanguageRecognizer         | SI     | 14.5367    | 22.6229     |
    | **SOTA_CSLR**                  | **SI** | **7.3123** | **13.0652** |

## Baseline Overview

To evaluate the performance of our proposed models, we compare them against baseline architectures implemented in this research. These models include established and recent approaches in sequence modeling, from classic recurrent networks to hybrid architectures incorporating Large Language Models (LLMs). The performance of each baseline on the Isharah-1000 Signer-Independent (SI) and Unseen-Sentences (US) tasks is reported:

- **LLM-SlowFast (`llm_advslowfast`):** This model implements the SlowFast [3] concept to pose data, with parallel transformer pathways processing the sequence at different temporal dimensions. It further inserts linguistic knowledge by concatenating features from a pretrained XLM-RoBERTa model [11] before the final classifier.

- **LLaMA-Former (`mixllama`):** This baseline uses a standard transformer encoder to process pose features, which are then fed into a frozen LLaMA-2 [35] model to act as a sequential processor. This approach explores leveraging the advanced sequence modeling capabilities of a large generative LLM.

- **LLaMA-SlowFast (`mixllama + slowfastllm`):** This model fuses LLaMA-2 and a SlowFast architecture to extract multi-rate temporal features from pose data. The fused visual features are then processed by an AraBERT model [8].

- **ST-GCN-Conformer (`gcn_transformer`):** This model first employs a Spatial-Temporal Graph Convolutional Network (ST-
  GCN) to learn features directly on the skeletal graph [37]. The output of the ST-GCN is then processed by a conformer encoder to capture the long-range sequential relationships among these learned spatio-temporal features.

- **DistilBERT-Former (`LLM Backbone (DistilBERT)`):** This model initially processes the pose sequence using a standard transformer encoder to capture visual-temporal dependencies. The resulting feature embeddings are then fed into a pretrained Distil-BERT model [33]. This approach aims to leverage the linguistic and contextual knowledge inherent in the LLM backbone.

- **Mamba-Sign (`MambaSignLanguageRecognizer`):** A hybrid Mamba-transformer block is utilized in this architecture, replacing traditional attention based backbones. This design leverages the linear-time sequence modeling strengths of Mamba and the global context capabilities of self-attention [17]. It represents an exploration of recent state-space models for their effectiveness in handling long sequences.

- **BiLSTM (`LSTM`):** This is a classic CSLR baseline consisting of a simple Bi-directional Long Short-Term Memory (BiL-LSTM) network [20]. It processes the pose features directly to capture temporal dependencies.

- **Sign-Conformer (`SignLanguageConformer`):** This network modifies the conformer architecture, which has shown great success in sign language domain. It combines convolutions and self-attention to capture both local and global dependencies in the pose sequence [18].

- **CNN-BiLSTM (`SignLanguageRecognizer`):** This architecture combines a a Temporal Convolutional Network (TCN) [26]with a BiLSTM backbone. The convolutional layers extract and downsample local spatio-temporal features, which are then modeled by the BiLSTM to capture long-range dependencies.
