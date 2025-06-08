# VaeDiff-DocRE
Source code for the paper: "[VaeDiff-DocRE: End-to-end Data Augmentation for Document-level Relation Extraction via Variational Autoencoder and Diffusion Prior](https://aclanthology.org/2025.coling-main.488/)", has been accepted at COLING 2025.

Please follow instructions in `README` of each folder for setting up environment.

Trained models for Stage 2 and Stage 3 are [here](https://drive.google.com/file/d/1EHDDvZewVaTEZ7iebPrGoilTOo8u1g7R/view?usp=sharing).

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

Contact person: **Khai Tran**, [phankhai.tran@uq.edu.au](mailto:phankhai.tran@uq.edu.au)

## Citation
If you find our work useful, please cite our work as:
```bibtex
@inproceedings{tran2025vaediff,
  title={VaeDiff-DocRE: End-to-end Data Augmentation Framework for Document-level Relation Extraction},
  author={Tran, Khai Phan and Hua, Wen and Li, Xue},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics},
  pages={7307--7320},
  year={2025}
}
```
> **Abstracts:**
> Document-level Relation Extraction (DocRE) aims to identify relationships between entity pairs within a document. However, most existing methods assume a uniform label distribution, resulting in suboptimal performance on real-world, imbalanced datasets. To tackle this challenge, we propose a novel data augmentation approach using generative models to enhance data from the embedding space. Our method leverages the Variational Autoencoder (VAE) architecture to capture all relation-wise distributions formed by entity pair representations and augment data for underrepresented relations. To better capture the multi-label nature of DocRE, we parameterize the VAE’s latent space with a Diffusion Model. Additionally, we introduce a hierarchical training framework to integrate the proposed VAE-based augmentation module into DocRE systems. Experiments on two benchmark datasets demonstrate that our method outperforms state-of-the-art models, effectively addressing the long-tail distribution problem in DocRE.

## Acknowledgement
The work partially uses released code from following papers. Thanks author for their works!
```
[1] Tan, Q., He, R., Bing, L., & Ng, H. T. (2022). Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation. Findings of the Association for Computational Linguistics: ACL 2022.
[2] Guo, J., Kok, S., & Bing, L. (2023). Towards Integration of Discriminability and Robustness for Document-Level Relation Extraction. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics. Association for Computational Linguistics.
[3] Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In International conference on machine learning (pp. 8162-8171). PMLR.
[4] Lovelace, J., Kishore, V., Wan, C., Shekhtman, E., & Weinberger, K. Q. (2023). Latent diffusion for language generation. Advances in Neural Information Processing Systems, 36, 56998-57025.
```