---
Multiview Garment Modeling:
---

how to setup:

- `bash setup.sh` to setup the environment

how to run:

- `python pretrain.py` train the model first on consistent views

- `python train.py` for training

- `python eval.py` for testing

---
Main Goal of this project:
---

- Disentanglement: if we perform some edits on the back side of a shirt, it should not affect the front side as much as it changes the back-view.

- Iterative Addition: We iteratively add strokes to a sketch, that should also improve folds in the shirt.

---
Question:
---

- How is it garment-specific?

---
Contributors:
---

pinakinathc, tuangfeng, yulia, duygu, song

### Track experiments

- `model_A.py` -- Proposed method with Binary mask
- `model_AA.py` -- Baseline method with Binary mask but first pass feat
- `model_B.py` -- Baseline method with Sigmoid attention
- `model_BB.py` -- Baseline method with Sigmoid but first pass feat
- `model_C.py` -- Baseline method with Cross-Attention Binary Mask
- `model_D.py` -- Baseline method with Max. Pooling
- `model_E.py` -- Baseline method with Avg. Pooling
- `model_F.py` -- Baseline method with Attention Pooling
- `model_G.py` -- Baseline method with Concatenation
- `model_H.py` -- Baseline method with RNN