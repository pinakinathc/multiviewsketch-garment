# Garment Ideation: Iterative View-Aware Sketch-Based Garment Modeling

### **Authors**

**[Pinaki Nath Chowdhury](https://pinakinathc.me), [Tuanfeng Wang](https://tuanfeng.github.io/), [Duygu Ceylan](https://www.duygu-ceylan.com/), [Yi-Zhe Song](https://scholar.google.co.uk/citations?user=irZFP_AAAAAJ&hl=en), [Yulia Gryaditskaya](https://yulia.gryaditskaya.com/)**


SketchX, Center for Vision Speech and Signal Processing

University of Surrey, United Kingdom

Adobe Research London, UK

Surrey Institute for People Centred AI

**Published at 3DV 2022 ORAL**

[[Paper]](http://www.pinakinathc.me/assets/papers/3DV_2022.pdf) [[Supplemental]](http://www.pinakinathc.me/assets/papers/3DV_2022_supp.pdf)


### **Abstract**
Designing real and virtual garments is becoming extremely demanding with rapidly changing fashion trends and increasing need for synthesizing realistically dressed digital humans for various applications. However, traditionally designing real and virtual garments has been time-consuming. Sketch based modeling aims to bring the ease and immediacy of drawing to the 3D world thereby motivating faster iterations. We propose a novel sketch-based garment modeling framework that is specifically targeted to synchronize with the iterative process of garment ideation, e.g., adding or removing details from different views in each iteration. At the core of our learning based approach is a view-aware feature aggregation module that fuses the features from the latest sketch with the thus far aggregated features to effective refine the generated 3D shape. We evaluate our approach on a wide variety of garment types and iterative refinement scenarios. We also provide comparisons to alternative feature aggregation methods and demonstrate favorable results. 


### **Our envisioned User Interface**
![Envisioned-User-Interface](http://www.pinakinathc.me/assets/images/3DV22-teaser.png)

### **How to cite this paper**
```
@inproceedings{chowdhury2022garment,
    title={Garment Ideation: Iterative View-Aware Sketch-Based Garment Modeling}
    author={Chowdhury, Pinaki Nath and Wang, Tuanfeng and Ceylan, Duygu and Song, Yi-Zhe and Gryaditskaya, Yulia},
    booktitle={3DV},
    year={2022}
}
```

## Instructions

how to setup:

- `bash setup.sh` to setup the environment

how to run:

- `python train.py --model_name=model_AA --exp_name=model_AA --data_dir=<path/to/dataset>`
