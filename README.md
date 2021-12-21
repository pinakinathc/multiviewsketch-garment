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

### Few observations

[17 December 2021]

* starting with partial training from scratch leads to open meshes during generation. This is likely because the model never saw a full shirt and has no idea how it should look like.

* I am training on both Siggraph15 dataset and adobe dataset together.

* Focusing more on the normal setup (without partial supervision) for the old and new updater network