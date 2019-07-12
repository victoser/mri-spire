# MRI-SpIRe - the MRI Sparse Iterative Reconstructor

This package contains a simple iterative MRI reconstruction algorithm based on Compressed Sensing.
It reconstructs a series of MR images from undersampled k-space data acquired through a qMRI protocol.

For a theoretical background in Sparse Reconstruction and some examples of the functionality of the 
package, please check the included demo in demo/Demo.ipynb. For detailed instructions, read the documentation.

This work incorporates elements of existing algorithms for Compressed Sensing reconstruction in quantitative MRI<sup>1-4</sup>.

## References
1. Mariya Doneva, Peter Börnert, Holger Eggers, Christian Stehning, Julien Sénégas, and Alfred Mertins. Compressed sensing reconstruction for magnetic resonance parameter mapping. Magnetic Resonance in Medicine, 64(4):1114–1120, 6 2010.
2. Mariya Doneva and Alfred Mertins. MRI: Physics, Image Reconstruction, and Analysis. chapter 3, pages 51–71. CRC Press, 2015.
2. Michael Lustig, David L Donoho, Juan M Santos, and John M Pauly. Compressed sensing MRI. IEEE SIGNAL PROCESSING MAGAZINE, 25(2):72–82,3 2008.
3. Chuan Huang, Christian G Graff, Eric W Clarkson, Ali Bilgin, and Maria I Altbach. T2 mapping from highly undersampled data by reconstruction of principal component coefficient maps using compressed sensing. MAGNETIC RESONANCE IN MEDICINE, 67(5):1355–1366, 5 2012.
