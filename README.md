# Food Classification with Deep Learning in fastai

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qwhsq4_uHJcKUvrYU6Ly7CfAvBTrEOnl?usp=sharing)

## Introduction

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. 

Convolutional Neural Networks (CNNs), a technique within the broader Deep Learning field, have been a revolutionary force in Computer Vision applications, especially in the past half-decade or so. One main use-case is that of image classification, e.g. determining whether a picture is that of a dog or cat.

![CNN Image](https://missinglink.ai/wp-content/uploads/2019/07/A-Convolutional-Neural-Network.png)

Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. This enables CNNs to learn complex functions and they can easily be scaled to thousands of different classes, as seen in the well-known ImageNet dataset of 1000 classes, used as a benchmark for computer vision algorithm performance. 

In the past couple of years, these cutting edge techniques have started to become available to the broader software development community. Industrial strength packages such as PyTorch and Tensorflow have given us the same building blocks that Google uses to write deep learning applications for embedded/mobile devices to scalable clusters in the cloud -- *Without having to hand-code the GPU matrix operations, partial derivative gradients, and stochastic optimizers that make efficient applications possible.*

On top of all of this, are user-friendly APIs such as fastai and Keras that abstract away some of the lower level details and allow us to focus on rapidly prototyping a deep learning computation graph. 

## Project Description

As an introductory project for myself, I chose to use a pre-trained image classifier and retrain it on a dataset that I find interesting, the Food-101 dataset.

![Food-101-sample-image](https://camo.githubusercontent.com/a169296b1c9b506d8cfd76c8099a62a915e6febc/68747470733a2f2f7777772e766973696f6e2e65652e6574687a2e63682f64617461736574735f65787472612f666f6f642d3130312f7374617469632f696d672f666f6f642d3130312e6a7067)

In the future, I want to also make something to be able to predict the amount of food and give the nutritional values based on that, but for now, lets do the image classification part. 

## Dataset

Food-101 is a challenging dataset consisting of 101,000 images of 101 different food classes and 1000 images for each class with images of upto 512 x 512 resolution. Every class has 750 training and 250 test images. 

The dataset contains the following classes: 

```python
apple_pie	    eggs_benedict	     onion_rings
baby_back_ribs	    escargots		     oysters
baklava		    falafel		     pad_thai
beef_carpaccio	    filet_mignon	     paella
beef_tartare	    fish_and_chips	     pancakes
beet_salad	    foie_gras		     panna_cotta
beignets	    french_fries	     peking_duck
bibimbap	    french_onion_soup	     pho
bread_pudding	    french_toast	     pizza
breakfast_burrito   fried_calamari	     pork_chop
bruschetta	    fried_rice		     poutine
caesar_salad	    frozen_yogurt	     prime_rib
cannoli		    garlic_bread	     pulled_pork_sandwich
caprese_salad	    gnocchi		     ramen
carrot_cake	    greek_salad		     ravioli
ceviche		    grilled_cheese_sandwich  red_velvet_cake
cheesecake	    grilled_salmon	     risotto
cheese_plate	    guacamole		     samosa
chicken_curry	    gyoza		     sashimi
chicken_quesadilla  hamburger		     scallops
chicken_wings	    hot_and_sour_soup	     seaweed_salad
chocolate_cake	    hot_dog		     shrimp_and_grits
chocolate_mousse    huevos_rancheros	     spaghetti_bolognese
churros		    hummus		     spaghetti_carbonara
clam_chowder	    ice_cream		     spring_rolls
club_sandwich	    lasagna		     steak
crab_cakes	    lobster_bisque	     strawberry_shortcake
creme_brulee	    lobster_roll_sandwich    sushi
croque_madame	    macaroni_and_cheese      tacos
cup_cakes	    macarons		     takoyaki
deviled_eggs	    miso_soup		     tiramisu
donuts		    mussels		     tuna_tartare
dumplings	    nachos		     waffles
edamame		    omelette
```

Some of the classes are just variants of the same kind of food, which makes it very hard to differentiate between even for a human. For example, the only difference between steak and filet mignon is where from the cattle's body is the meat sourced from. 

![Steak and filet mignon look the same!](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/efdb7e4d-3b88-4c6d-958a-bb4deb405262/imgonline-com-ua-twotoone-hSfiSA9Bte1.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T071951Z&X-Amz-Expires=86400&X-Amz-Signature=0d39156d1252a9f65dc5a25be4af6baacf16327562065a16c1c7025810d377dd&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22imgonline-com-ua-twotoone-hSfiSA9Bte1.jpg%22)

Figure 1 : Steak (left) and Filet Mignon (right)

Even in the same class, images can vary wildly. For example, all of the images in Figure 2 have been labelled as "bread pudding", yet even as a human, I think I’d struggle to classify them as such.

![Figure 2: A sample of images from the Food-101 dataset, all labelled as “bread pudding”.](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bdf300f2-e7f7-45ae-ad69-9577c690dfd1/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T072021Z&X-Amz-Expires=86400&X-Amz-Signature=898c970fb9871a2b798d3800bb862d7cd8575085b669f28cdf06cb1cf5a464ea&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

Figure 2: A sample of images from the Food-101 dataset, all labelled as “bread pudding”.

The creators of the dataset left the training images deliberately uncleaned, so there are a number of mislabeled images, and as we can see, a large range in brightness / color saturation. More fundamentally, however, it’s clear that no two bread puddings are quite alike (or at all alike it seems). Classifying images with such high intra-class variation is hard.

If our model is to perform well “in the wild”, it needs to be able to handle these issues: real, non-professional photos of food aren’t going to be of uniform size and are unlikely to have perfect lighting or framing.

If you’re just interested in the results go to the Results section.

## Data augmentation and dataloaders

Data augmentation is a strategy that significantly increases the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks.

I have used fastai's inbuilt data augmentation tools here:

```python
tfms = get_transforms(do_flip=False,        #randomly flip the image horizontally
                      max_rotate=20.,       #randomly rotate images by upto 20 degrees
                      max_zoom=1.25,        #random zoom
                      max_warp=0.1)         #random perspective warping 
```

Next, we create a ImageList using fastai's `ImageList.from_folder` class.

```python
src =  (ImageList.from_folder(path=path).
        split_by_rand_pct(0.2).
        label_from_folder())

data = src.transform(tfms,size=128).databunch(bs=16).normalize(imagenet_stats)
```

Lets look at one image from each class: 

![one image from each class](https://miro.medium.com/max/2000/1*y1NL-R2kj9NvEwi3dSPmwg.png)

## The model

### Residual Networks

A residual neural network (ResNet) is a modifications to CNN that builds on constructs known from pyramidal cells in the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers. Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities (ReLU) and batch normalization in between. 

![Residual Networks diagram](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6b0644bc-d248-46f0-b8c6-ccb7857a1a76/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T072127Z&X-Amz-Expires=86400&X-Amz-Signature=7c61e771564be74f44b3d6dbcda9da5f295090cf8ff0866e4c38348e1940af7e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

I am using the ResNet-50 architecture with pretrained weights on the ImageNet dataset containing ~14 million images. 

![ResNet-50 architecture](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/07306487-a7ed-4bf9-9e3b-9fdd2fc88961/model2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T072129Z&X-Amz-Expires=86400&X-Amz-Signature=3cd30c0310a05ad49aa952f2e15a100333c8770af816facd3d825fccd1d75ba7&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22model2.png%22)

The ResNet-50 architecture.

```python
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
```

### Progressive resizing

Progressive resizing is a technique for building CNNs that can be very helpful during the training and optimization phases of a machine learning project. The technique appears most prominently in Jeremy Howard’s work, and he uses it to good effect throughout his terrific [fast.ai](http://fast.ai/) course, “Deep Learning for Coders”.

We use progressive resizing by first overfitting on 128 x 128 images and progressively scaling up to 256 x 256 and 512 x 512 images. 

### learn.lr_find()

learn.lr_find is an amazing function in the fastai library which tries a range of learning rates and is a massive help in choosing the proper the learning rate hyper parameter. 


```python
learn.lr_find()
learn.recorder.plot()
```

![lr find diagram](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8f7e267e-8d0f-4376-920f-b04ccd8e765f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T072224Z&X-Amz-Expires=86400&X-Amz-Signature=9c9b2094b138cabd377b42c739f4db7c7c3f4b20840f6e05ce14015a279bf13e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## Results

After about 35 epochs of training (12,14,9 on 128, 256 and 512 pixel images respectively), we achieve a accuracy of **87 percent** before we start to overfit**.** This is pretty impressive for such a difficult dataset. 

### Plotting top losses

We can plot the images with the maximum error along with their predicted and actual classes using the ClassificationInterpretation class from fastai. Let's have a look at the 25 images our model got the most wrong.

```python
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
interp.plot_top_losses(25, figsize=(25,25))
```
![25 Top losses image](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4b0c2ef3-1d5f-4471-bf0b-cad6a8c8515b/ssfefe.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T072621Z&X-Amz-Expires=86400&X-Amz-Signature=f9ab887e3eb4de7fc00affcf0139049d1064bc903a63cb4a917e30b8f9deeaee&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22ssfefe.png%22)

![25 Top Losses image](https://github.com/anirudha-akela/food-app/blob/master/images/lrfind.png)


As we can see, a lot of the errors are on food items that are quite similar and hard to distinguish between like chocolate cake and strawberry shortcake.

![mislabeled pizza detected correctly!!](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7201556b-752a-4e58-88cd-794dd9c812d8/pizza.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T072254Z&X-Amz-Expires=86400&X-Amz-Signature=a24847982a7c7b23e022734074d84c8e8208dca339b215689eb3df50caa22400&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22pizza.png%22)

In one of the images, the ground truth is actually mislabeled, but still our model manages to predict the correct class!!

### Confusion matrix

Lets plot the confusion matrix to see what all our model is getting wrong. 

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_cm(y_true, y_pred, figsize=(100,100)):
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=data.classes, columns=data.classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "Blues", annot=annot, fmt='', ax=ax , linewidths = 0.2, cbar = False)
    
plot_cm(y, pred_class)
```
![Confusion matric with percentages](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6be68d8e-6550-4ebd-b3d6-c2802e7db6ad/final_confusion_matrix.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200630T073225Z&X-Amz-Expires=86400&X-Amz-Signature=2128b5538997a7b99b7ffa6bc500304a07cd4a4d7f9512d7977dc9d6285cd834&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22final_confusion_matrix.png%22)
Such a large number of classes mean that the confusion matrix is very sparse and thus it is very difficult to gather any useful information from it. Thankfully, fastai has us covered once again. 

### Most confused classes

Fastai's aptly named most_confused function returns all the classes from whom more than a certain number of images were confused. Lets look at all the classes in which more than 10 images were wrongly predicted. (this is still only 4% of the per class validation dataset).

```python
interp.most_confused(min_val=10)
```

```python
[('steak', 'filet_mignon', 30),
 ('chocolate_mousse', 'chocolate_cake', 26),
 ('donuts', 'beignets', 22),
 ('apple_pie', 'bread_pudding', 21),
 ('chocolate_cake', 'chocolate_mousse', 21),
 ('filet_mignon', 'steak', 21),
 ('prime_rib', 'steak', 14),
 ('pork_chop', 'filet_mignon', 13),
 ('pork_chop', 'grilled_salmon', 13),
 ('tuna_tartare', 'beef_tartare', 13),
 ('dumplings', 'gyoza', 12),
 ('lobster_bisque', 'clam_chowder', 12),
 ('pork_chop', 'steak', 12),
 ('ravioli', 'lasagna', 12),
 ('steak', 'prime_rib', 11),
 ('tiramisu', 'chocolate_mousse', 11),
 ('bread_pudding', 'apple_pie', 10),
 ('frozen_yogurt', 'ice_cream', 10),
 ('ravioli', 'gnocchi', 10),
 ('tuna_tartare', 'ceviche', 10)]
```

Predictably, we see a lot of confusion between foods of the same general classes (chocolate_mousse/chocolate_cake , steak/filet_mignon/prime_rib/pork_chop, tuna_tartare/beef_tartare). This problem nicely highlights the challenges in food classification as we face a lot of classes that are very similar visually and may differ on other factors such texture or presentation which are not captured easily by CNNs. 

### Links to code:

[GitHub](https://github.com/anirudha-akela/food-app)

[Colab Notebook](https://colab.research.google.com/drive/1Qwhsq4_uHJcKUvrYU6Ly7CfAvBTrEOnl?usp=sharing)

[Jovian Notebook](https://jovian.ml/anirudha-akela/food101-fastai)

## References

1. [Food-101 – Mining Discriminative Components with Random Forests](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
2. ResNet-50 paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 
3. Food-101 dataset: [https://vision.ee.ethz.ch/datasets_extra/food-101/](https://vision.ee.ethz.ch/datasets_extra/food-101/)
4. Food-101 dataset hosted by fastai:  [https://course.fast.ai/datasets](https://course.fast.ai/datasets)
5. [New Food-101 SoTA with fastai and platform.ai’s fast augmentation search](https://platform.ai/blog/page/3/new-food-101-sota-with-fastai-and-platform-ais-fast-augmentation-search/)
6. Papers with Code, [SoTA of Image Classification on Food-101](https://paperswithcode.com/sota/fine-grained-image-classification-on-food-101)
7. [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
