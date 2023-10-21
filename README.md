# Autoencoders and Variational Autoencoders (VAE) Playground Repository

This repository contains projects I played with related to various types of autoencoders/
Below, you will find an overview of the different autoencoder types, their benefits, and details about the projects available in this repository.



## Built with

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange?style=flat&logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.4%2B-red?style=flat&logo=keras)](https://keras.io/)


## Autoencoders

Autoencoders are neural networks designed for data compression and dimensionality reduction. They consist of two main parts: an encoder and a decoder. The encoder maps the input data to a lower-dimensional representation, and the decoder reconstructs the original data from this representation. <br>
Autoencoders can have various architectures, and two common types are Convolutional Autoencoders (ConvAE) and Variational Autoencoders (VAE).

## Convolutional Autoencoders (ConvAE)

Convolutional Autoencoders are a type of autoencoder specifically designed for processing image data. They use convolutional layers to efficiently capture spatial patterns in the input images. Benefits of ConvAE include:

- Effective feature extraction from images.
- Dimensionality reduction for image compression.
- Image denoising capabilities by learning to remove noise during reconstruction.

### Projects Inside:

1. **ImageCompressionPytorchConvAE:** This project focuses on using ConvAE for image compression, where the network learns to represent images in a compact form while preserving important information.

2. **ImageDenoisingConvAutoEncoders:** In this project, ConvAE is used for image denoising. It learns to remove noise from noisy images while retaining image details.

3. **ComparisonConvAE_FeedFowardAE.ipynb:** A Jupyter notebook that compares ConvAE with Feedforward Autoencoders (FeedFowardAE) to highlight the advantages of ConvAE for image-related tasks.

4. **OtherAutoencoders:**

   - **Recurrent AE:** A Recurrent Autoencoder designed for sequential data, which can capture temporal dependencies in the input.
   - **Conv AE:** A traditional Convolutional Autoencoder for image-related tasks, similar to ConvAE but with a different implementation.

5. **Undercomplete AutoEncoder Tabular Dataset:** This project demonstrates the use of undercomplete autoencoders on tabular data, showcasing how autoencoders can be applied beyond image data.



## Variational Autoencoders (VAE)

Variational Autoencoders are a type of autoencoder that introduces probabilistic elements. VAEs are useful for generating new data samples and have benefits like:

- Improved data generation and interpolation.
- The ability to model complex data distributions.
- Encoding data into a continuous latent space.

### Projects Inside:

1. **VariationalAutoEncoder:** This project explores the concept of VAE and its applications, including data generation from a learned latent space.

2. **ConvVariationalAutoEncoder.ipynb:** A Jupyter notebook that demonstrates how VAE can be used for image compression while preserving generative capabilities.

## Usage

Feel free to explore the projects within this repository and use the provided code and notebooks for your own experiments and applications.

## License


For more information and usage instructions for individual projects, please refer to their respective directories.

If you have any questions or issues, don't hesitate to reach out or create a new issue in the repository.

Happy coding!

## Contact:
Feel free to reach out to me on LinkedIn or through email & don't forget to visit my portfolio.
 
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/samiabelhaddad/)
  [![Email](https://img.shields.io/badge/Email-Contact%20Me-brightgreen?style=flgat&logo=gmail)](mailto:samiamagbelhaddad@gmail.com)
  [![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20My%20Portfolio-white?style=flat&logo=website)](https://your-portfolio-url-here.com/)
