# CMRI_orient

In this paper, we address the challenge of cardiac MRI imaging orientation and propose a framework based on deep neural networks for classifying and standardizing orientation. We propose a transfer learning strategy for multiple sequences and modalities of MRI, which adapts our model from a single modality to multiple modalities. For the orientation classification network, we compare various convolutional networks and select a lightweight three-layer network as the backbone. We embed the orientation classification network into a command-line tool for cardiac MRI orientation adjustment to achieve orientation correction for 3D NIFTI images.

