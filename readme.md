

## Problem Definition 🥅

This project is a passion project that contributes to building a free photo restore website where users can upload their photos and choose to modify the photo in 2 ways:

1. Colourise their photo. If their photo is in B&W or if the colours in the photo are faded. A user can provide their photo as input and the website will add colour to the photo to bring it to life
2. Improve photo resolution. If a user’s photo is low resolution, they can provide their photo as input and the website will increase output the same photo in a higher resolution.

This project will focus on gaining user feedback and iterating over the end product to continuously improve its performance, usability and reliability.

## PSA 📣

At this {link}[https://sriyaroy.notion.site/Design-Doc-18dfa72bedf1800a9c25e0c0cb1a03e5?pvs=74], you will find the original plan of this project which is live on my Notion website. However, since beginning this project, the work and its purpose have changed as I've found new areas for learning!

## In this repo 📋

### Date: 28th july 2025

This repo is my work to build my own super resolution model from scratch after being inspired by the interesting research in this area. So far, I've built a basic U-net and trained this on roughly 3k images that I've sourced from popular super resolution datasets like DF2k and RealSRSet. I've recently implemented a tensorboard to visualise the loss curves and I have a strong feeling that overfitting is taking place based on the results on my validation images. There are a few areas to tackle this:
- Creating a loss function that is more accurate to what we'd like the model learn (Currenlty, I've kept things simple with a L2 MSE loss but _this is not a good choice for super resolution due to its tendency to encourage blurriness)
- Having some better metrics would be useful as well. So far I'm just watching my very basic loss go down which is not an accurate representation of the model's learning
- I also believe my model is too small for the number of data points so it will need to be made deeper. My intention is to make this deeper without creating a training slowdown by utilising skip connections as I have been doing so far.

I hope to update my progress here as the model gets better and better! 🎉

## Feedback welcome!
Get in touch with me by email at sriyaroy@yahoo.com 📩