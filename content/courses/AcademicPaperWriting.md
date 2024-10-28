---
title: "AcademicPaperWriting"
date: 2024-10-16
draft: false
tags:
    - courses
---

!This post is an introduction to [SNI-SLAM](https://github.com/IRMVLab/SNI-SLAM) from my "Academic Paper Writing" course.

# **SNI-SLAM: Semantic Neural Implicit SLAM**

![pic 1](/YulinBlog/images/courses/AcademicPaperWriting01.png#center)

yulinJoseph 2024.10.12

---

# **Table of Contents**

**1. Prerequisites**

&ensp;&ensp;&ensp;&ensp;What are SLAM and NeRF?

**2. Related Work**

&ensp;&ensp;&ensp;&ensp;Previous Approaches and (Dis)Advantages

**3. Methodology**

&ensp;&ensp;&ensp;&ensp;Hierarchical Semantic Representation, Cross-Attention,\
&ensp;&ensp;&ensp;&ensp;Internal Fusion-Based Decoder and Feature Loss

**4. Conclusion**

&ensp;&ensp;&ensp;&ensp;Performance and ?

![pic 2](/YulinBlog/images/courses/AcademicPaperWriting02.png#center)

---

# **01 Prerequisites**

## **SLAM: Simultaneous Localization and Mapping**

-   **Two key tasks:**
    -   **Localization:** Determining the precise position and orientation (such as x, y, z coordinates and the direction) of the robot or device in the environment in real-time.
    -   **Mapping:** Generating a real-time map of the environment surrounding the device, including the location and shape of obstacles in the environment.
-   **Main steps:**
    -   Sensor Data Collection, Feature Extraction and Matching, Position Estimation and Map Construction, Loop Closure Detection

---

# **01 Prerequisites**

## **SLAM**

![pic 3](/YulinBlog/images/courses/AcademicPaperWriting03.gif#center)

---

# **01 Prerequisites**

[**Video of NeRF**](https://www.bilibili.com/video/BV1va4y1r7Yg#center)

The core idea of NeRF is to use a neural network to represent the 3D space of a scene. In this scene, the position and viewing direction of each point are input into the network, and the network outputs the color (RGB) and volume density of that point. Through volume rendering techniques, NeRF can sample multiple 3D points along each ray (a ray projected from the camera into the scene), and finally generate the color for each pixel, synthesizing an image.

![pic 4](/YulinBlog/images/courses/AcademicPaperWriting04.png#center)

---

# **02 Related Work**

## **Why SLAM with NeRF?**

_"Traditional semantic SLAM has limitations including its inability to predict unknown areas and high storage space requirements."_

-   Advantages of Using NeRF:
    1. Continuous Scene Representation
    2. High-Quality Reconstruction
    3. Multi-Modal Information Fusion
    4. Low Storage Requirement
    5. Viewpoint Independence

---

# **02 Related Work**

## **Two Challenges of Semantic NeRF-based SLAM:**

1. Appearance, geometry and semantic information are interrelated, so processing them independently will lose interact connections, leading to an incomplete understanding of the image or scene.

    - MSeg3D[1] fuses geometry and semantic features to obtain more accurate semantic segmentation results. However, this work does not take advantage of appearance information as another modality to enhance semantic expression from the visual structural perspective. Moreover, mutual reinforcement of different modalities is not explored either.

2. As the appearance of a scene, such as color, varies under different views, leveraging semantic multi-view consistency to optimize appearance will affect the details of the appearance, and vice versa.

    - Semantic-NeRF[2] appends a segmentation renderer before injecting viewing directions into the Multi-layer Perceptron (MLP). However, the impact of semantic optimization on appearance and geometric expression is not explored.

---

# **02 Related Work**

-   **Semantic SLAM**
    -   SLAM++[3] is object-aware RGB-D SLAM that uses joint pose graph to represent object-level information in the scene.
    -   Kimera[4] relies on RGB-D or stereo sensing to generate dense semantic mesh maps and uses visual inertial odometry for the motion estimation.
-   **Neural implicit SLAM**
    -   iMAP[5] introduces a single MLP network to achieve real-time mapping and localization of the scene.
    -   NICE-SLAM[6] adopts hierarchical feature grid as scene representation, enabling more accurate mapping.
    -   ESLAM[7] uses multi-scale axis-aligned feature planes, reducing the memory consumption growth.
    -   Vox-Fusion[8] is based on octree management for incremental mapping.
    -   vMAP[9] is an object-level dense SLAM system that utilizes semantic segmentation results for object association, but it does not perform semantic mapping.
    -   NIDSSLAM[10] uses ORB-SLAM3[11] for tracking and InstantNGP[12] for mapping. For the processing of semantic information, it maps the segmentation results to color encodings for optimization of network. However, this work does not integrate semantic with other features of the environment, such as geometry and appearance.

---

# **03 Methodology**

## **An overview of SNI-SLAM**

![pic 5](/YulinBlog/images/courses/AcademicPaperWriting05.png#center)

---

# **03 Methodology**

## **1 Cross-Attention based Feature Fusion**

In this work, we utilize **an universal feature extractor Dinov2**[13], followed by segmentation head to construct the segmentation network. The extracted **semantic feature** lacks specificity to the environment as it is derived from a pretrained segmentation network. Therefore, we utilize real-time updated appearance MLP $H_{\theta}$ to transform the semantic feature into **appearance feature** $f_a=H_{\theta}\left(f_s\right)$(corrected, error in paper). This MLP network stores environment-specific appearance information. For **geometry feature**, we first obtain the coordinates of 3D points $\lbrace{p_i}\rbrace^{N}_{i=1}$ through ray sampling. Then, we use a NeRF-based frequency encoding[14] to get vector $\gamma\left(p\right)$:

$$\gamma\left(p\right)=\left(sin2^0\pi{p},cos2^0\pi{p},\dots,sin2^{L-1}\pi{p},cos2^{L-1}\pi{p}\right),$$

where $L$ defines the total count of frequencies used. We use $L=6$ for 3D coordinates. $\gamma\left(p\right)$ is processed through geometry MLP $E_\theta\left(\gamma\left(p\right)\right)$ to obtain geometry feature $f_g$, which stores geometry information of the environment.

---

# **03 Methodology**

## **1 Cross-Attention based Feature Fusion**

Then, we leverage the structural property of geometry to guide attention. $f_g$ is used as $Q$, $f_a$ is used as $K$, and $f_s$ is used as $V$, to perform cross-attention calculation to obtain fused semantic feature $T_s$:

$$T_s=softmax\left(\frac{f_g\cdot{f_a^T}}{\sqrt{\left \| f_a \right \|^2_2 } }\right)f_s.$$
(added a dot and modified the parentheses compared to paper)

Through this fusion, the weighted combination of semantic information is dynamically adjusted based on geometry and appearance feature matches, thereby minimizing the influence of incorrect semantic predictions by highlighting matches and downplaying mismatches. Moreover, we utilize $f_a$, $f_g$ and fused semantic features $T_s$ as $V$, $Q$ and $K$, to obtain fused appearance feature $T_a$ respectively:

$$T_a=softmax\left(\frac{f_g\cdot{T_s^{T}}}{\sqrt{\left \| T_s \right \|^2_2 } }\right)f_a.$$

---

# **03 Methodology**

## **1 Cross-Attention based Feature Fusion**

![pic 6](/YulinBlog/images/courses/AcademicPaperWriting06.png#center)

---

# **03 Methodology**

## **2 Hierarchical Semantic Mapping**

Currently, existing NeRF-based semantic modeling methods employ single-level neural implicit representation, regardless of whether they use voxel grid[15] or MLP[16, 17]. However, their performances are often limited when dealing with complex scenarios. We discover that using a hierarchical approach is more effective for semantic representation of the environment. When looking at a scene, we first grasp the overall layout and identify the main objects to develop a coarse understanding. After that, we shift our focus to more finely detailed. This top-down approach allows us to understand and process complex semantic information more naturally and efficiently. Therefore, we employ coarse-to-fine semantic modeling for scene representation in this paper. Moreover, we design a fusion-based decoder to obtain semantic, color, SDF values, then achieve semantic, RGB, depth images through rendering process.

\*SDF: Signed Distance Field, a function used to represent the geometry of a 3D scene implicitly by encoding the distance from any point in space to the nearest surface of an object.

---

# **03 Methodology**

## **2 Hierarchical Semantic Mapping**

**Coarse-to-fine Semantic Representation.** We utilize feature planes[18] to store features, which saves storage space compared with voxel grid[19, 20]. For semantic mapping, we employ a coarse-to-fine semantic representation. For each feature plane, we use two different levels of spatial resolution. For a given coordinate, we then concatenate the corresponding coarse and fine feature. We demonstrate empirically (?) that the introduction of multilevel semantic representations improves the performance of implicit semantic modeling and provides finer and richer semantic understanding.

![pic 7](/YulinBlog/images/courses/AcademicPaperWriting07.png#center)

---

# **03 Methodology**

## **2 Hierarchical Semantic Mapping**

**Decoder Design.** In our work, we incorporate the idea of feature collaboration into decoder module to obtain SDF, RGB, and semantic values from geometry, appearance and semantic features. Inside the decoder, we concatenate geometry feature with appearance and semantic features, then the concatenated feature passes through MLP network to obtain color decoding information. This design provides one-way correlation to ensure that improvement and application of the features occur only in one direction, thereby preventing mutual interference between the features. In addition, it also facilitates information exchange between features, improving the network’s understanding of them. Considering the complexity of rich semantic categories, a larger hidden layer is necessary for comprehensive modeling.

---

# **03 Methodology**

## **2 Hierarchical Semantic Mapping**

**Rendering.** We sample N points on the ray $\lbrace{p_n}\rbrace_{i=1}^N$ to generate color $c\left(p_n\right)$, semantic $s\left(p_n\right)$ and TSDF $d\left(p_n\right)$ values of these points through decoder $D_\theta\left(p_n\right)$. Then, we use the SDF-based rendering method proposed in StyleSDF [21] to convert SDF values into volume densities:

$$\sigma_g\left(p_n\right)=\frac{1}{\alpha_g}\cdot{sigmoid}\left(-\frac{d\left(p_n\right)}{\alpha_g}\right),$$

$$\sigma_s\left(p_n\right)=\frac{1}{\alpha_s}\cdot{sigmoid}\left(-\frac{d\left(p_n\right)}{\alpha_s}\right),$$

where $\alpha_g$ represents a learnable parameter that determines the level of sharpness along the surface boundary. Another learnable parameter $\alpha_s$ is used for semantic rendering.

Volume density $\sigma_g\left(p_n\right)$ is subsequently utilized in rendering both the color and depth associated with each ray to obtain rendered color $\hat{c}$ and depth $\hat{d}$:

$$w_g=\exp\left(-\sum_{i=1}^{n-1}\sigma_g\left(p_i\right)\right)\left(1-\exp\left(-\sigma_g\left(p_n\right)\right)\right),$$

$$\hat{c}=\sum^{N}\_{i=1}w_g\cdot{c\left(p_n\right)}, \hat{d}=\sum^{N}\_{n=1}w_g\cdot{z_n}.$$

In this context, $z_n$ represents the depth of point pn in relation to the camera’s pose. $\sigma_s\left(p_n\right)$ is used in semantic rendering and obtain rendered semantic $\hat{s}$:

$$w_s=\exp\left(-\sum_{i=1}^{n-1}\sigma_s\left(p_i\right)\right)\left(1-\exp\left(-\sigma_s\left(p_n\right)\right)\right),$$

$$\hat{s}=\sum^{N}_{i=1}w_s\cdot{s\left(p_n\right)}.$$

---

# **03 Methodology**

## **3 Loss Functions**

**Semantic Loss.** For the supervision of semantic information, we use cross-entropy loss. It is worth noting that in the process of rendering semantics, we detach the gradient to prevent the semantic loss from interfering with the optimization of geometry and appearance features:

$$\mathcal{L}\_s=-\sum\_{m\in{M}}\sum^{L}_{l=1}p_l\left(m\right)\cdot\log\hat{p_l}\left(m\right),$$

where $p_l$ represents multi-class semantic probability at class $l$of the ground truth map.

**Feature Loss.** When only using color, depth, and semantic values as supervision signals, the MLP network will overly focus on less significant details and ignore some more salient features. To address this problem, feature loss is constructed and utilized to provide additional guidance for updating feature plane and MLP network. By providing direct supervision on intermediate features, this higher-level loss enables the scene representation to preserve important details:

$$\mathcal{L}=\left \| f\_\mathrm{extract}-f\_\mathrm{interp} \right \|\_1.$$

where $f_\mathrm{extract}$ represents the feature map, $f_\mathrm{interp}$ stands for features obtained by the interpolation from the feature planes. The extracted features are more accurate and used as supervision signals.

**Color and Depth Loss.** The input is RGB-D frames containing ground truth RGB and depth values. We construct color and depth loss by comparing the rendered RGB and depth values with the ground truth values. These loss functions are then utilized for updating the network:

$$\mathcal{L}\_c=\frac{1}{|M|}\sum^{|M|}\_{i=0}\left \| C_i-C^{gt}_{i} \right \|,$$

$$\mathcal{L}\_d=\frac{1}{|M|}\sum^{|M|}\_{i=0}\left \| D_i-D^{gt}_{i} \right \|,$$

where $C_i$, $D_i$ are rendered RGB and depth values, $C_i^{gt}$, $D_i^{gt}$ are ground truth values.

---

# **04 Conclusion**

## **Qualitative comparison**

![pic 8](/YulinBlog/images/courses/AcademicPaperWriting08.png#center)

---

# **04 Conclusion**

## **Qualitative comparison**

![pic 9](/YulinBlog/images/courses/AcademicPaperWriting09.png#center)

$\text{Depth L1} = \frac{1}{N} \sum_{i=1}^{N} |D_{\text{pred},i} - D_{\text{gt},i}|$

$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \min_{p_j^{\text{gt}} \in S^{\text{gt}}} \|\mathbf{p}_i^{\text{pred}} - \mathbf{p}_j^{\text{gt}}\|$ (Accuracy)

$\text{Comp} = \frac{1}{M} \sum*{i=1}^{M} \min*{p_j^{\text{pred}} \in S^{\text{pred}}} \|\mathbf{p}\_i^{\text{gt}} - \mathbf{p}\_j^{\text{pred}}\|$ (Completeness)

$\text{Comp. Ratio} = \frac{1}{M} \sum_{i=1}^{M} \mathbb{1}\left[ \min_{p_j^{\text{pred}} \in S^{\text{pred}}} \|\mathbf{p}_i^{\text{gt}} - \mathbf{p}_j^{\text{pred}}\| < \delta \right]$ (Completion Ratio)

$\text{ATE Mean} = \frac{1}{N} \sum\_{i=1}^{N} \|\mathbf{p}\_i^{\text{pred}} - \mathbf{p}\_i^{\text{gt}}\|$ (Absolute Trajectory Error Mean)

$\text{ATE RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\mathbf{p}_i^{\text{pred}} - \mathbf{p}_i^{\text{gt}}\|^2}$ (Absolute Trajectory Error Root Mean Square Error)

---

# **04 Conclusion**

## **Ablation Study**

![pic 10](/YulinBlog/images/courses/AcademicPaperWriting10.png#center)

$\text{IoU}(c) = \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$ (Intersection over Union)

$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \text{IoU}(c)$ (Mean Intersection over Union)

---

# **04 Conclusion**

https://github.com/MLNLP-World/Paper-Writing-Tips

---

# **References**

---
