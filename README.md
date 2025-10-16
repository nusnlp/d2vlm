
<p align="center">
  <h1 align="center">Factorized Learning for Temporally Grounded Video-Language Models</h1>
  <p align="center">
    <a href="https://wenzhengzeng.github.io/">Wenzheng Zeng</a><sup>1</sup>,
<!--     · -->
    <a href="https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en">Difei Gao</a><sup>1</sup>,
<!--     · -->
    <a href="https://scholar.google.com/citations?&user=h1-3lSoAAAAJ&hl=en">Mike Zheng Shou</a><sup>1</sup>,
<!--     · -->
    <a href="https://scholar.google.com/citations?user=FABZCeAAAAAJ&hl=en">Hwee Tou Ng</a><sup>1</sup>,
    
  </p>
  <p align="center"><sup>1</sup>National University of Singapore</p>
  <h3 align="center">ICCV 2025</h3>

  <h3 align="center"> 
  <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_Factorized_Learning_for_Temporally_Grounded_Video-Language_Models_ICCV_2025_paper.pdf">📄 Paper</a> &nbsp; | &nbsp;
  <a href="https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zeng_Factorized_Learning_for_ICCV_2025_supplemental.pdf">📄 Supp.</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/wenzhengzeng/D2VLM-Models">🤗 Model</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/datasets/wenzhengzeng/D2VLM-Dataset">🤗 Dataset</a> &nbsp; | &nbsp;
  <a href="https://iccv.thecvf.com/media/PosterPDFs/ICCV%202025/1301.png?t=1759990615.5445755">🖼️ Poster</a> &nbsp; | &nbsp;
  <a href="https://www.youtube.com/watch?v=DylkFjyTITs&t=2s">▶️ Video</a> &nbsp; | &nbsp;
</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
    <img src="pictures/fig1.png" width="100%"/>

</p>
This repository contains the official implementation of the ICCV 2025 paper "Factorized Learning for Temporally Grounded Video-Language Models".


## 🔆 Highlights 

- **Model:** We propose a new framework $D^2\mathrm{VLM}$, where we decompose the generation objective into a "grounding then answering with evidence referencing" paradigm and introduce evidence tokens to emphasize explicit event-level visual semantic capture.
- **Training Algorithm:** We introduce Factorized Preference Optimization (FPO) that explicitly addresses both temporal grounding and textual response. A factorized data synthesis approach is also designed to support FPO.
- **Performance:** Our method consistently outperforms SOTA methods across various tasks.
- **Open Source:** We release the source code and model weights to the community.


## 🔥News
* [2025-10] Code and model weights are released!
* [2025-06] Our work is accepted to ICCV 2025!


## 🛠️ Installation

Please refer to the following environmental settings that we use. You may install these packages by yourself if you meet any problem during automatic installation.

- CUDA 11.8
- Python 3.12.2
- PyTorch 2.4.0
- [Transformers](https://github.com/huggingface/transformers) 4.44.2
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) 0.14.5
- [NNCore](https://github.com/yeliudev/nncore) 0.4.5

### Install from source

1. Clone the repository from GitHub.

```shell
git clone https://github.com/nusnlp/d2vlm.git
cd d2vlm
```

2. Initialize conda environment.

```shell
conda create -n d2vlm python=3.12 -y
conda activate d2vlm
```

3. Install dependencies.

```shell
pip install -r requirements.txt
```

## 📦 Dataset

Please refer to the [Dataset](https://huggingface.co/datasets/wenzhengzeng/D2VLM-Dataset) page.

## 🤖 Inference and Evaluation
1. You can download the pre-trained model at [here](https://huggingface.co/wenzhengzeng/D2VLM-Models).
2. Run the following commands. Remember to change the absolute path within each .sh file.

### E.T. Bench

```shell
  bash scripts/inference.sh

  # or refer to the inference and Evaluation part of scripts/train_inference_eval.sh
```


### Charades-STA
```shell
  bash all_benchmark_eval/charades/inference.sh
```

all_benchmark_eval/charades/inference.sh
### Youcook2

```shell
  bash all_benchmark_eval/youcook2/inference.sh
```


## 💪 Training
1. Download the pretrained model from [here](https://huggingface.co/PolyU-ChenLab/ETChat-Phi3-Mini-Stage-2) (stage-2 model of E.T. Chat).
2. Check and run the following command (modify relevant path).
```shell
  bash scripts/train_inference_eval.sh
```



## 🎓 Citation

If you find our work useful in your research, please consider to cite our paper:

  ```
  @inproceedings{d2vlm,
    title={Factorized Learning for Temporally Grounded Video-Language Models},
    author={Zeng, Wenzheng and Gao, Difei and Shou, Mike Zheng and Ng, Hwee Tou},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025},
    pages={20683-20693}
  }
  ```
  
## 🙏 Acknowledgments

This project was built upon [E.T. Bench](https://github.com/PolyU-ChenLab/ETBench), [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat), and [AMP](https://github.com/takomc/amp). We thank their solid contribution to the community!
