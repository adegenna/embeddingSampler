![unittests](https://github.com/adegenna/embeddingSampler/actions/workflows/unittests.yml/badge.svg)

Library for tools related to sampling from a linear embedded subspace. Allows user to define a target distribution in an ambient space (with box constraints) of arbitrary dimension, then draw samples from an embedded linear subspace which map to that target distribution. See https://arxiv.org/abs/2001.11659 for details, as this is part of the ALEBO algorithm for high dimensional Bayesian inference.

Copyright 2021 by Anthony M. DeGennaro (ISC License).

**Installation**

```sh
cd [/PATH/TO/embeddingSampler]
pip install ./
```

**Unit Tests**

```sh
cd [/PATH/TO/embeddingSampler]
pytest -o log_cli=true -rP
```

**Example Driver**

```sh
python -c "import embsamp.examples.example_2to3d; embsamp.examples.example_2to3d.main()"
```

![embedded_space](https://user-images.githubusercontent.com/2964258/131155586-dd6a70ad-edfb-428b-a4a7-e2973365cf38.png)
![ambient_space](https://user-images.githubusercontent.com/2964258/131155601-57ed4dc2-2f94-41a7-ae8e-77a859690c19.png)
