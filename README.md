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

If you use this repository or would like to refer to the paper, please use the following:

```bash
@ARTICLE{Ning2024adversarial, 
  author={Zhu, Ning and Wang, Jing and Zhang, Ying and Wang, Huan and Han, Te},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={An Adversarial Training Framework Based on Unsupervised Feature Reconstruction Constraints for Crystalline Silicon Solar Cells Anomaly Detection}, 
  year={2024},
  volume={73},
  number={},
  pages={1-13},
  keywords={Photovoltaic cells;Training;Feature extraction;Image reconstruction;Silicon;Anomaly detection;Inspection;Anomaly detection;electroluminescence (EL) imaging;generative adversarial networks (GANs);unsupervised learning},
  doi={10.1109/TIM.2024.3462989}
}
