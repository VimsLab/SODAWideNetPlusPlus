# SODAWideNetPlusPlus \[[Link](https://arxiv.org/pdf/2408.16645?)\]
Combining Attention and Convolutions for Salient Object Detection

### ABSTRACT

Salient Object Detection (SOD) has traditionally relied on feature refinement modules that utilize the features of an ImageNet pre-trained backbone. However, this approach limits the possibility of pre-training the entire network because of the distinct nature of SOD and image classification. Additionally, the architecture of these backbones originally built for Image classification is sub-optimal for a dense prediction task like SOD. To address these issues, we propose a novel encoder-decoder-style neural network called SODAWideNet++ that is designed explicitly for SOD. Inspired by the vision transformers' ability to attain a global receptive field from the initial stages, we introduce the Attention Guided Long Range Feature Extraction (AGLRFE) module, which combines large dilated convolutions and self-attention. Specifically, we use attention features to guide long-range information extracted by multiple dilated convolutions, thus taking advantage of the inductive biases of a convolution operation and the input dependency brought by self-attention. In contrast to the current paradigm of ImageNet pre-training, we modify 118K annotated images from the COCO semantic segmentation dataset by binarizing the annotations to pre-train the proposed model end-to-end. Further, we supervise the background predictions along with the foreground to push our model to generate accurate saliency predictions. SODAWideNet++ performs competitively on five different datasets while only containing 35{\%} of the trainable parameters compared to the state-of-the-art models. 

## Model Overview

| Model | Number of Parameters | Pre-computed Saliency Maps | Model Weights | Pre-trained Weights |
|------------|----------------------|----------------------------|---------------|---------------|
| SODAWideNet++    | 26.58M                | [Saliency Maps](https://drive.google.com/drive/folders/12ZpJ5aewFwlX_avG-RNHBWPKaqXTZvIO?usp=sharing) | [Weights](https://drive.google.com/file/d/1HY3zRTmZ58g9RCo8PDza6p0Kg0x5POLF/view?usp=sharing) | [Pre-trained Weights](https://drive.google.com/file/d/1gsDc77dNn2lFAMbnnteKhJjNoG3ZXSbe/view?usp=sharing) |
| SODAWideNet++-M   | 6.66M                | [Saliency Maps](https://drive.google.com/drive/folders/1zdf1j8xsPnSWS3xK2HAMoz2u9MJoK8iT?usp=sharing) | [Weights](https://drive.google.com/file/d/10kwnn9asoUeZSsjIuZQMKjm_XPNrv3i0/view?usp=sharing) | Pre-trained Weights |
| SODAWideNet++-S    | 1.67M                | [Saliency Maps](https://drive.google.com/drive/folders/1_MtnVz1qU63AlrlZ98fCu51gNVyMDnWA?usp=sharing) | [Weights](https://drive.google.com/file/d/1RRRIUYAZU_8Si0yNPvSCimz1m3kEIBIe/view?usp=sharing) | [Pre-trained Weights](https://drive.google.com/file/d/1xu0eOa51ufxiGM4gq1b0ay9WAi_Y_J9b/view?usp=sharing) |

## COCO Pre-training

Download the dataset and unzip the file. Then, use the following command to train the model. The model sizes can be **L**, **M**, and **S**.

```bash
python training.py \
    --lr 0.001 \
    --epochs 21 \
    --f_name "COCOSODAWideNet++L" \
    --n 4 \
    --b 20 \
    --sched 1 \
    --training_scheme "COCO" \
    --salient_loss_weight 1.0 \
    --use_pretrained 0 \
    --im_size 384 \
    --model_size 'L'
```

## DUTS Finetuning

Download the dataset from [link](https://drive.google.com/file/d/1-sxp99YoDRSQBebMWXLeI0tlkRsU_LrH/view?usp=sharing) and unzip the file. Then, use the following command to train the model. Create a folder with the name **checkpoints** and save the COCO pre-trained checkpoint in it. 

```bash
python training.py \
    --lr 0.001 \
    --epochs 11 \
    --f_name "DUTSSODAWideNet++L" \
    --n 4 \
    --b 20 \
    --sched 1 \
    --training_scheme "DUTS" \
    --salient_loss_weight 0.5 \
    --use_pretrained 1 \
    --checkpoint_name "COCOSODAWideNet++L"
    --im_size 384 \
    --model_size 'L'
```

## Inference

We provide an option to generate the saliency map for a single image or multiple images in a folder. The below script displays the generated saliency map. **model_size** can be **L**, **M**, and **S**.
```bash
python inference.py \
    --mode single \
    --input_path /path/to/image.jpg \
    --display \
    --model_size L
```

The below script generates a saliency map and saves the result.
```bash
python inference.py \
    --mode single \
    --input_path /path/to/image.jpg \
    --model_size L
```
The below script generates saliency maps for a folder of images and saves them in the user-specified output directory.
```bash
python inference.py \
    --mode folder \
    --input_path /path/to/input/folder \
    --output_dir /path/to/output/folder \
    --model_size L
```

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


