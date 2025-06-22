# Adversarial Unsupervised Anomaly Detection for Crystalline Silicon Solar Cells

This repository provides the official implementation of the IEEE TIM paper:

**"An Adversarial Training Framework Based on Unsupervised Feature Reconstruction Constraints for Crystalline Silicon Solar Cells Anomaly Detection"**  
by *Ning Zhu, Jing Wang, Ying Zhang, Huan Wang, and Te Han*  
Published in **IEEE Transactions on Instrumentation and Measurement**, 2024.

Key innovations include:
- ✨ **Dual-space feature reconstruction loss** to enforce both visual fidelity and representational coherence
- 📐 **GCT (Gaussian Context Transformer)** blocks for enhanced long-range dependency modeling
- ⚙️ Encoder-Decoder-Encoder adversarial architecture for powerful representation learning

To set up the environment, please install dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
