# SODAWideNetPlusPlus \[[Link](https://arxiv.org/pdf/2408.16645?)\]
Combining Attention and Convolutions for Salient Object Detection

### ABSTRACT

Salient Object Detection (SOD) has traditionally relied on feature refinement modules that utilize the features of an ImageNet pre-trained backbone. However, this approach limits the possibility of pre-training the entire network because of the distinct nature of SOD and image classification. Additionally, the architecture of these backbones originally built for Image classification is sub-optimal for a dense prediction task like SOD. To address these issues, we propose a novel encoder-decoder-style neural network called SODAWideNet++ that is designed explicitly for SOD. Inspired by the vision transformers' ability to attain a global receptive field from the initial stages, we introduce the Attention Guided Long Range Feature Extraction (AGLRFE) module, which combines large dilated convolutions and self-attention. Specifically, we use attention features to guide long-range information extracted by multiple dilated convolutions, thus taking advantage of the inductive biases of a convolution operation and the input dependency brought by self-attention. In contrast to the current paradigm of ImageNet pre-training, we modify 118K annotated images from the COCO semantic segmentation dataset by binarizing the annotations to pre-train the proposed model end-to-end. Further, we supervise the background predictions along with the foreground to push our model to generate accurate saliency predictions. SODAWideNet++ performs competitively on five different datasets while only containing 35{\%} of the trainable parameters compared to the state-of-the-art models. 

## Model Overview

| Model | Number of Parameters | Pre-computed Saliency Maps | Model Weights |
|------------|----------------------|----------------------------|---------------|
| SODAWideNet++    | 26.58M                | [Saliency Maps](https://drive.google.com/drive/folders/12ZpJ5aewFwlX_avG-RNHBWPKaqXTZvIO?usp=sharing) | Weights | 
| SODAWideNet++-M   | 6.66M                | [Saliency Maps](https://drive.google.com/drive/folders/1zdf1j8xsPnSWS3xK2HAMoz2u9MJoK8iT?usp=sharing) | Weights |
| SODAWideNet++-S    | 1.67M                | [Saliency Maps](https://drive.google.com/drive/folders/1_MtnVz1qU63AlrlZ98fCu51gNVyMDnWA?usp=sharing) | Weights |

## COCO Pre-training

Download the dataset and unzip the file. Then, use the following command to train the model.

#### Citation

If you find our research useful, please cite our paper with the following citation - 

```
@InProceedings{10.1007/978-3-031-78192-6_14,
    author="Dulam, Rohit Venkata Sai and Kambhamettu, Chandra",
    title="SODAWideNet++: Combining Attention and Convolutions for Salient Object Detection",
    booktitle="Pattern Recognition",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="210--226",
    isbn="978-3-031-78192-6"
}
```


