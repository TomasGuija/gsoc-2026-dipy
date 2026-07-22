---
layout: page
title: "A Summer of Scientific Software: My GSoC Journey with DIPY"
description: "My Google Summer of Code 2026 work on new registration metrics, a faster SyN backend, robust SynthSeg utilities, and reproducible evaluation."
author: "Tomás Guija-Valiente"
---

<div align="center">
  <p>
    <img src="./docs/GSoC.svg" alt="Google Summer of Code" width="410">
  </p>
  <p>
    <img src="./docs/PSF.png" alt="Python Software Foundation" width="260" style="margin: 0 18px;">
    <img src="./docs/dipy-logo.png" alt="DIPY — Diffusion Imaging in Python" width="260" style="margin: 0 18px;">
  </p>
</div>

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    },
    displayAlign: 'center'
  });
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-MML-AM_CHTML"></script>

I am Tomás Guija-Valiente, a computer vision and machine learning researcher at the medical imaging laboratory [PROMISE–LAIMBIO](https://www.linkedin.com/company/promise-laimbio/). My academic background combines mathematics, computer science, and artificial intelligence. Over the past year, I have become increasingly interested in applying these skills to the medical domain, where research can have a meaningful real-world impact and open-source software plays a particularly important role.

[Google Summer of Code 2026](https://summerofcode.withgoogle.com/) gave me the opportunity to work at this intersection. This year, I joined [DIPY](https://dipy.org/) through the [Python Software Foundation](https://python-gsoc.org/), hoping not only to implement a new algorithm or feature, but also to learn how a mature scientific library is designed, tested, reviewed, and maintained by its community.

My project, [**Adding New Similarity Metrics for DIPY's Image Registration Frameworks**](https://github.com/satwiksps/GSoC_archive_2026/blob/main/Python%20Software%20Foundation/Accepted/DIPY.pdf), focused on medical image registration: the process of estimating a spatial transformation that aligns two or more images. Registration is a fundamental component of longitudinal analysis, atlas construction, and population studies, but its performance depends heavily on how image similarity is defined and how efficiently the transformation can be optimized.

I began the summer with one main goal: adding Mutual Information to DIPY’s deformable registration framework. As the project progressed, I also worked on improving performance, testing the implementations, building better benchmarks, and refining related APIs. Overall, the project became a valuable introduction to mathematical implementation, scientific software development, and collaborative open-source work, with the guidance and support of my mentors, [Serge](https://sergekoudoro.com/) and [Atharva](https://atharva-shah-2298.github.io/). Eight pull requests and many commits later, this report brings together the main results of that experience: the new algorithms, performance improvements, fixes, and lessons learned along the way.

## Summary

| Area | Contribution | Pull request |
|---|---|---|
| Deformable registration | Mutual Information metric for SyN with dense, local-support derivatives | [#4067](https://github.com/dipy/dipy/pull/4067) |
| Affine registration | Local cross-correlation metric compatible with `AffineRegistration` | [#4078](https://github.com/dipy/dipy/pull/4078) |
| SyN performance | Threaded field composition, fixed-point inversion, and image warping | [#4073](https://github.com/dipy/dipy/pull/4073), [#4083](https://github.com/dipy/dipy/pull/4083) |
| MI backend | Removed a redundant cubic B-spline evaluation from Parzen histograms | [#4071](https://github.com/dipy/dipy/pull/4071) |
| Metric correctness | Fixed stale energy state after the backward CC step | [#4047](https://github.com/dipy/dipy/pull/4047) |
| SynthSeg | Added cleaner optional brain masks and flexible model-space dimensions | [#4052](https://github.com/dipy/dipy/pull/4052), [#4080](https://github.com/dipy/dipy/pull/4080) |
| Evaluation | Built a reproducible, SLURM-based comparison pipeline for DIPY and ANTs | [Benchmarking branch](https://github.com/TomasGuija/dipy/tree/mi-benchmarking/benchmarks/registration) |

## 1. Mutual Information for SyN registration

**Pull request:** [#4067 — MI implementation for SyN registration](https://github.com/dipy/dipy/pull/4067)

DIPY already provided Mutual Information (MI) for affine registration, while its [SyN implementation](https://docs.dipy.org/stable/interfaces/registration_flow.html#symmetric-diffeomorphic-registration) supported cross-correlation, sum of squared differences, and expectation-maximization metrics. The main objective of my project was to add MI to the deformable framework, where it can support multimodal problems in which corresponding anatomy does not necessarily have similar intensity values.

### Mutual information

Mutual Information asks a simple question: **how much does knowing the intensity in one image tell us about the intensity in the other?** For fixed- and moving-image intensities $F$ and $M$, it is written as

$$
I(F;M)
=
\sum_{i,j} p_{FM}(i,j)
\log\!\left(\frac{p_{FM}(i,j)}{p_F(i)p_M(j)}\right)
$$

Here, $p_{FM}(i,j)$ describes how often intensity $i$ in the fixed image appears together with intensity $j$ in the moving image. The terms $p_F(i)$ and $p_M(j)$ describe how common those intensities are individually. If the two images are unrelated, their joint distribution is close to the product of the two individual distributions and MI is low. As the images become better aligned, their intensity relationship becomes more predictable and MI increases.

This is particularly useful for multimodal registration. A structure may appear bright in a T1-weighted MRI and dark in a B0 image, so comparing the intensities directly would be unreliable. MI does not require them to be equal; it only looks for a consistent relationship between them.

### Parzen windows

In practice, these probability distributions are estimated with a joint histogram. A standard histogram uses hard assignments: each sample falls into exactly one bin. That creates abrupt jumps—a tiny change in intensity can move a sample from one bin to another—which is a problem for a gradient-based registration algorithm that needs a smooth signal telling it how to update the transformation.

DIPY follows the [Mattes MI formulation](https://doi.org/10.1109/TMI.2003.809072) and uses [Parzen windows](https://doi.org/10.1214/aoms/1177704472) to soften those assignments. On the moving-image axis, each sample spreads its contribution across four neighboring bins using a cubic B-spline. As the intensity changes, the weights move smoothly between those bins instead of jumping abruptly. The derivative of the B-spline weights then tells the optimizer how a small intensity change would affect MI.

### Implementation details

DIPY's existing affine MI implementation produces a derivative for a small set of transformation parameters. SyN is different: it optimizes a displacement vector at every voxel. A direct extension of the affine approach would require storing derivative information for every histogram bin, voxel, and spatial dimension, which would use far too much memory.

The new implementation avoids that large intermediate representation by exploiting the local support of the Parzen kernel. For each voxel, it:

1. identifies the fixed-image bin and the four moving-image bins affected by the sample;
2. computes how the weights of those four bins change with moving-image intensity;
3. combines that change with the spatial gradient of the moving image;
4. accumulates the result directly into the voxel's 2D or 3D displacement update.

This produces the dense vector field expected by `SymmetricDiffeomorphicRegistration` without constructing a histogram-by-voxel derivative tensor. The low-level work is implemented in Cython alongside DIPY's existing Parzen-histogram code, with support for 2D and 3D `float32` and `float64` images.

At the Python level, the PR adds an `MIMetric` class that follows the same lifecycle as DIPY's other SyN metrics: initialize the current iteration, compute forward and backward updates, and report the scalar energy. The design was informed by the local-support path in [ITK's Mattes MI metric](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1MattesMutualInformationImageToImageMetricv4.html).

### Validation

I validated the implementation on the [MRBrainS18 dataset](https://mrbrains18.isi.uu.nl/), available through [DataverseNL](https://doi.org/10.34894/E0U32Q). The initial benchmark covered 100 inter-patient, cross-modality registrations using the dataset's FLAIR, inversion-recovery, and T1-weighted images and manual segmentations. The complete preparation and evaluation protocol, together with all benchmarking code, is available in the [`mi-benchmarking` branch](https://github.com/TomasGuija/dipy/tree/mi-benchmarking).

The table reports the mean and standard deviation across the 100 registrations:

| Method | NCC | NMI | Label Dice | Label Jaccard |
|---|---:|---:|---:|---:|
| Rigid baseline | 0.362 ± 0.086 | 1.021 ± 0.006 | 0.506 ± 0.056 | 0.369 ± 0.051 |
| ANTs SyN | **0.561 ± 0.087** | **1.044 ± 0.013** | 0.565 ± 0.049 | 0.430 ± 0.046 |
| DIPY SyN | 0.549 ± 0.084 | 1.043 ± 0.012 | **0.567 ± 0.051** | **0.432 ± 0.048** |

The improvement over the rigid baseline confirms that the new DIPY MI metric produces effective multimodal registration updates.

## 2. Local cross-correlation for affine registration

**Pull request:** [#4078 — CC implementation for affine registration](https://github.com/dipy/dipy/pull/4078)

The second metric contribution adds local cross-correlation (CC) to DIPY's affine API. While MI is useful when the images come from different modalities, CC is a natural choice for images from the same modality, where the same anatomy is expected to produce similar local intensity patterns.

### Local cross-correlation

Instead of comparing individual pixels or voxels, local CC compares small neighborhoods. Before comparing them, it subtracts the average intensity in each neighborhood and normalizes by their local variation. This makes the metric respond to the shape of the intensity pattern rather than to simple differences in brightness or contrast.

For a neighborhood $W_x$ around voxel $x$, the squared local correlation is

$$
\operatorname{CC}(x)=
\frac{
\left[\sum_{y\in W_x}(F(y)-\bar F_x)(M(y)-\bar M_x)\right]^2
}{
\left[\sum_{y\in W_x}(F(y)-\bar F_x)^2\right]
\left[\sum_{y\in W_x}(M(y)-\bar M_x)^2\right]
}
$$

In simple terms, the numerator measures whether both neighborhoods vary together, while the denominator accounts for how much each one varies on its own. The score becomes high when their local patterns match. This is the local CC formulation used in the [SyN work by Avants et al.](https://doi.org/10.1016/j.media.2007.06.004) and already available in DIPY's deformable registration backend.

### Implementation details

DIPY already had optimized code for computing local CC statistics and voxel-level updates for SyN. The main challenge was adapting that machinery to affine registration, where the optimizer changes a small set of global parameters instead of a displacement vector at every voxel.

The new metric:

1. converts the optimizer's current parameter vector into an affine matrix;
2. warps the moving image onto the fixed-image grid and computes the local CC factors;
3. evaluates the moving-image gradient at the transformed fixed-grid points;
4. uses a Cython kernel to combine the local CC factor, image gradient, and transform Jacobian directly into the gradient for each transform parameter.

At the API level, `CrossCorrelationMetric` follows the same optimizer-facing structure as DIPY's [existing affine MI metric](https://docs.dipy.org/stable/reference/dipy.align.html#dipy.align.imaffine.MutualInformationMetric), implementing `setup`, `distance`, `gradient`, and `distance_and_gradient`.

### Validation

In a representative 3D b0-to-b0 registration, global NCC improved at every stage:

| Stage | Global NCC |
|---|---:|
| Identity | 0.1903 |
| Center of mass | 0.4772 |
| Translation | 0.5584 |
| Rigid | 0.6656 |
| Affine | 0.6751 |

The consistent improvement confirms that the metric works correctly with DIPY's affine optimization pipeline.

## 3. Faster SyN kernels with OpenMP

SyN spends much of its time repeatedly combining and inverting displacement fields and warping images. Since many pixels or voxels can be processed independently, these operations are good candidates for multithreading. I addressed them in two related pull requests.

### 3.1 Field composition and fixed-point inversion

**Pull request:** [#4073 — Parallelize displacement-field composition](https://github.com/dipy/dipy/pull/4073)

The outer composition loops now use Cython's [`prange`](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html): rows are distributed across workers in 2D and slices in 3D. Each worker writes to a disjoint region of the output field.

The serial kernel also accumulated composition statistics. To avoid data races, the threaded implementation stores per-thread sample counts, sums, higher-order sums, and maxima, then merges them after the parallel region. The same `num_threads` configuration is propagated into fixed-point inversion, which repeatedly calls field composition.

Serial and threaded tests compare both the composed fields and the returned statistics. The following preliminary end-to-end SyN comparison used the same input images and a multiresolution iteration schedule of 20, 20, and 10:

| Method | Time (s) | NCC |
|---|---:|---:|
| DIPY serial | 301.155 | 0.95494 |
| DIPY threaded | 243.676 | 0.95494 |

Threading produced a **1.24× speed-up** over serial DIPY while preserving the same NCC. The complete implementation used for this experiment is available in the [`speed-syn` branch](https://github.com/TomasGuija/dipy/tree/speed-syn).

### 3.2 Image warping

**Pull request:** [#4083 — Multithreading 2D and 3D image warping](https://github.com/dipy/dipy/pull/4083)

The follow-up PR parallelizes `warp_2d`, `warp_2d_nn`, `warp_3d`, and `warp_3d_nn`, covering linear and nearest-neighbor interpolation. Dedicated row and slice helpers give each worker its own interpolation buffer. `DiffeomorphicMap` and `SymmetricDiffeomorphicRegistration` forward the optional thread count while preserving compatibility with existing calls.

Tests compare one-thread and multi-thread results across both dimensions, both interpolation modes, and forward and inverse map operations. A small 3D SyN timing experiment produced the following results:

| Revision and mode | Time (s) | NCC |
|---|---:|---:|
| Four threads | 76.864 | 0.95741 |
| DIPY `master` | 110.465 | 0.95741 |

Using four threads reduced runtime while preserving the final NCC.


## 4. Two focused registration fixes

### 4.1 Removing a redundant Parzen evaluation

**Pull request:** [#4071 — Avoid redundant Parzen spline-bin evaluation](https://github.com/dipy/dipy/pull/4071)

While developing MI for SyN, I noticed that the histogram loop checked five moving-bin offsets: `[-2, -1, 0, 1, 2]`. The `-2` offset is always at least two bins away from the continuous bin coordinate, where the cubic B-spline weight is zero. In practice, only `[-1, 0, 1, 2]` can contribute.

The change removes that evaluation from PDF and gradient computations and replaces hard-coded radius values with the existing `padding` variable. 

### 4.2 Correct energy state after backward CC updates

**Pull request:** [#4047 — Update CCMetric energy after backward step](https://github.com/dipy/dipy/pull/4047)

`CCMetric.compute_backward()` computed the backward energy but did not assign it to `self.energy`. Direct API users could therefore receive a stale or unset value from `get_energy()`.

The fix makes the backward path consistent with the forward path and with other metrics. Tests for `CCMetric` and `SSDMetric` now verify energy reporting after independent forward and backward computations. The issue was silent in the standard SyN workflow and was not expected to alter its registration result, but the fix makes the metric contract reliable for direct and custom use.

## 5. Robust SynthSeg integration

[SynthSeg](https://doi.org/10.1016/j.media.2023.102789) is designed to segment brain scans across contrasts and resolutions without retraining. Two contributions improve how DIPY exposes its masks and prepares model inputs.

### 5.1 Consistent and configurable brain masks

**Pull request:** [#4052 — Remove holes and islands from SynthSeg masks](https://github.com/dipy/dipy/pull/4052)

A mask obtained directly from predicted labels can contain small disconnected components or internal holes. This PR proposes making brain-mask generation a consistent part of `SynthSeg.predict()`, with a `finalize_mask` option that uses DIPY's existing `remove_holes_and_islands` function to clean those artifacts. Users can disable this step when they need the direct foreground mask given by `labels > 0`.

The PR also standardizes the returned labels, label dictionary, and mask, and forwards the same option through `BrainMaskFlow`. It explicitly distinguishes probability maps in SynthSeg's model space from masks recovered to the original image space, and adds tests for raw and finalized masks, probability outputs, shapes, and dtypes.

### 5.2 Flexible model-space dimensions

**Pull request:** [#4080 — Flexible input for SynthSeg](https://github.com/dipy/dipy/pull/4080)

DIPY previously forced every resampled input into a fixed $192\times192\times192$ model space, which could crop images with a larger field of view. The proposed preprocessing instead:

1. resamples the image to 1 mm isotropic resolution;
2. retains the complete resampled field of view;
3. pads each spatial dimension to the smallest multiple of 32;
4. recovers labels and masks in the original input space.

For a resampled dimension of length $s$, the compatible size is

$$
s_{\mathrm{pad}}=32\left\lceil\frac{s}{32}\right\rceil.
$$

This matches the [reference SynthSeg implementation](https://github.com/BBillot/SynthSeg), where cropping is optional and any requested crop size must be divisible by 32. Visual checks on OASIS-2 samples showed that the flexible path preserved anatomy that the fixed-size preprocessing could exclude.

## 6. Reproducible evaluation against ANTs

Alongside the DIPY changes, I developed a registration benchmark to compare DIPY with [ANTs](https://github.com/ANTsX/ANTs) under a shared protocol. The pipeline provides:

- common rigid prealignment;
- DIPY and ANTs SyN backends;
- monomodal and multimodal metrics;
- nearest-neighbor warping for anatomical labels;
- NCC and normalized MI for image similarity;
- Dice and Jaccard overlap for labels;
- one result file per fixed–moving pair;
- SLURM job arrays;
- aggregation into summary JSON files.

The benchmark is designed to compare the **SyN optimization stage itself**, rather than complete pipelines with different preprocessing or initialization. Both frameworks receive the same prepared images and rigid prealignment, and their settings are matched as closely as their interfaces allow. The accompanying documentation includes an in-depth comparison of the DIPY and ANTs implementations, their parameter conventions, and the decisions required to make the experiments comparable.

The original code and framework studies are available in the [`registration-benchmark` branch](https://github.com/TomasGuija/dipy/tree/registration-benchmark), which is now deprecated. The newer [`mi-benchmarking` branch](https://github.com/TomasGuija/dipy/tree/mi-benchmarking) extends the pipeline with Mutual Information as a SyN metric, although it currently needs to be rebased before further experiments are run.


## Conclusion

Taken together, these contributions strengthened several layers of DIPY's registration ecosystem, from similarity metrics and numerical kernels to preprocessing and evaluation.

The principal outcomes are:

- a dense local-support Mutual Information formulation for SyN;
- a local cross-correlation metric for affine registration;
- multithreaded composition, inversion, and warping kernels;
- two focused registration fixes already merged into DIPY;
- more robust SynthSeg mask and input-size handling; and
- a reproducible framework for comparing DIPY and ANTs.

Beyond the pull requests, commits, and lines of code, the most valuable outcome of this project has been the experience itself. It is remarkable how much can be learned in just over two months. This was my first real introduction to open-source development, and it showed me that implementing an idea is only the beginning.

Writing code has never been more accessible, especially with the tools available today. Contributing to a mature open-source project is a completely different challenge. The real work lies in producing code that is readable, well documented, robust, scalable, tested, and consistent with a well-established codebase used by thousands of people. Understanding that difference—and experiencing the care, discussion, and review required to reach that standard—has been one of the most important lessons of this project.

For that reason, the end of GSoC does not feel like an ending. I see it as the beginning of what I hope will be a long and rewarding journey contributing to free and accessible open-source software. I plan to continue supporting the contributions that remain under review, working with DIPY, and participating more broadly in the open-source community to help build useful tools that anyone can access.

I am especially grateful to my mentors, [Serge Koudoro](https://sergekoudoro.com/), [Atharva Shah](https://atharva-shah-2298.github.io/) and [Jong Sung Park](https://pjsjongsung.github.io/), for their guidance, patience, thoughtful reviews, and trust throughout the project. I also want to thank the wider DIPY community for welcoming me and making this first experience in open source such a meaningful one.

## Project links and references

- [DIPY website](https://dipy.org/) and [source repository](https://github.com/dipy/dipy)
- [DIPY registration documentation](https://docs.dipy.org/stable/interfaces/registration_flow.html)
- [My GSoC introduction and project overview](https://dipy.org/posts/2026/2026_05_25_Tomas.html)
- [My DIPY pull requests](https://github.com/dipy/dipy/pulls?q=is%3Apr+author%3ATomasGuija)
- [Google Summer of Code](https://summerofcode.withgoogle.com/) and [PSF GSoC](https://python-gsoc.org/)

## Personal links
- [LinkedIn](https://www.linkedin.com/in/tomas-guija-valiente)
- [GitHub](https://github.com/TomasGuija/)
