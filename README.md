# All-GAN-in-Keras

I want to create a repository to summarize what I have learnt from my experiments in which I use the Kears based GAN. I hope you also can find your needed information from this repository. 

# Why is keras not Pytorch

I know the Pytorch is more powerful and popular than Keras. However, I use the Keras in my working life. 

# What is GAN

I think you must know it. If not, you would not find me. 

# Where should I start my journey

Let we start from the DCGAN which is the one of the most popular GAN. 

## DCGAN
Deep Convolutional Generative Adversarial Network

paper: https://arxiv.org/abs/1511.06434 

![ezgif.com-gif-maker](https://i.imgur.com/jabOfBc.gif).

### Getting better results - CGAN

As you can see, the images are fuzzy and unmeaningful. This is because there are ten categories and we input is the noise with the same density. The output will fuse the images of the different categories. To address this problem, let we input the categories. 

Conditional Generative Adversarial Nets.

paper: https://arxiv.org/abs/1411.1784

![ezgif.com-gif-maker (1)](https://i.imgur.com/BwMtZpd.gif)

Since the original version is implemented by the Dense layers (full-connection), the output image is noisy. Now, we change to Conv layers.

![ezgif.com-gif-maker (2)](https://i.imgur.com/7RgnDDs.gif)

How about this result? More worst? THis is because the fusion step of the label and noise is implemented by *multiply*. Let we change to Concatenate layer

![ezgif.com-gif-maker (3)](https://i.imgur.com/Wh8M1A7.gif)

We have update to the multi-GPU implementation! However, it becomes much slower and the results is not improved.



# Is there any other GAN I can try?