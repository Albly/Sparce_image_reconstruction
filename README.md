# Image reconstruction: sparse linear inverse problem

## General Description

Sparse Linear Inverse Problem is the estimation of an unknown signal from indirect, noisy, underdetermined measurements by exploiting the knowledge that the signal has many zeros. We compare various iterative algorithmic approaches to this problem and explore how they benefit from loop-unrolling and deep learning.

In this project we've implemented deep neural networks for the sparse linear inverse problem for images in the wavelet domain, where the images are quite sparse. We are to reconstruct images distorted by white gaussian noise.

Applications: image compression, image quality enhancement (noise removal).
NLA methods: compressive sensing, sparse matrices processing and its storage, singular value decomposition, orthogonal transformations (DFT, DWT, DCT). 

## Data

1) TAMPERE17 noise-free image database.
https://webpages.tuni.fi/imaging/tampere17/

300 color images 512x512 pixels without noise (variance < 1), without interpolation and without lossy compression.

2) USC-SIPI dataset
https://www.kaggle.com/luffyluffyluffy/the-uscsipi-image-database

Only images with size 512x512 were left.

To reproduce the results and work with datasets you should download 2 datasets (archives) and then run all cells in datasets notebook.

## Files and folders description

Dataset.ipynb

Metrics.ipynb

main.ipynb 



## References

[1] M. Borgerding, P. Schniter, en S. Rangan, “AMP-Inspired Deep Networks for Sparse Linear Inverse Problems”, IEEE Transactions on Signal Processing, vol 65, no 16, bll 4293–4308, Aug 2017.

[2] N. Li en C. C. Zhou, “AMPA-Net: Optimization-Inspired Attention Neural Network for Deep Compressed Sensing”, CoRR, vol abs/2010.06907, 2020.

[3]  J. Jin, B. Yang, K. Liang, en X. Wang, “General image denoising framework based on compressive sensing theory”, Computers & Graphics, vol 38, bll 382–391, 2014.

[4] H. R. Shahdoosti en S. M. Hazavei, “A new compressive sensing based image denoising method using block-matching and sparse representations over learned dictionaries”, Multimedia Tools and Applications, vol 78, no 9, bll 12561–12582, Mei 2019.

[5] C. A. Metzler, A. Maleki and R. G. Baraniuk, "From Denoising to Compressed Sensing," in IEEE Transactions on Information Theory, vol. 62, no. 9, pp. 5117-5144, Sept. 2016, doi: 10.1109/TIT.2016.2556683.

[6] G. Nie en Y. Zhou, “ATP-Net: An Attention-based Ternary Projection Network For Compressed Sensing”, arXiv [eess.SP]. 2021.





add description 

add results

