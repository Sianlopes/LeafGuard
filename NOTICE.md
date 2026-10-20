# Data and reference attribution

PlantVillage: Mohanty, S. P., Hughes, D. P. and Salathé, M. (2016),
*Using Deep Learning for Image-Based Plant Disease Detection*,
https://doi.org/10.3389/fpls.2016.01419.
Color data: https://huggingface.co/datasets/mohanty/PlantVillage,
commit `9e97599868962bd0079b8db4b7f1efa9185fa1e7`, CC BY-SA 3.0.
The three small images in `data/examples/` are real, unmodified source test
images, selected before evaluation. Their original filenames and labels are
recorded in `data/examples/manifest.json`; they retain CC BY-SA 3.0. The full
raw collection is not distributed in this repository. Any included Grad-CAM example
is a derivative visualization of the source image identified in
`reports/integration.json`; retain PlantVillage attribution and CC BY-SA 3.0
when reusing that visualization. See https://creativecommons.org/licenses/by-sa/3.0/.

UCI Leaf: Silva, P. and Maral, A. (2013), *Leaf*, UCI Machine Learning Repository,
https://doi.org/10.24432/C53C78, CC BY 4.0.
Only RGB specimens from unsupported species are used for rejection checks.
Images are downloaded locally and are not distributed here. See
https://creativecommons.org/licenses/by/4.0/.

CIFAR-10: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny
Images* (2009), https://www.cs.toronto.edu/~kriz/cifar.html.
The pinned `uoft-cs/cifar10` mirror supplies source train/test object images for
separate negative collections; raw data are not distributed here.

EfficientNet: Tan, M. and Le, Q. V. (2019), https://arxiv.org/abs/1905.11946.
The transfer-learning model starts from the official Keras ImageNet weights.
TensorFlow/Keras and other dependencies retain their respective licenses.

Agricultural disease guidance is paraphrased from the references linked in
`data/disease_info.json`; it contains no guaranteed treatments or pesticide doses.
Full dataset checksums and negative-collection provenance are in the generated
metadata and `reports/ood_provenance.json`.
