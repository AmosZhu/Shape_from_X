# Shape_from_X
Mainly this repository is for education purpose and will contain some examples of 3D reconstruction with different methods. Including both DNN and classical methods.

Also, I am trying to migrate something I used daily into python package.
It will include utilities functions like I used for pytorch and some notebooks for easy explanation.

### Notebooks

#### Interpolations

- [x] Radial basis function

#### Multiple view Geometry

- [x] Structure from motion
- [ ] Calibration from Checkerboard
- [x] Auto calibration
- [ ] Trifocal tensor estimation

#### Neural Scene

- [ ] Nerf
- [ ] Barf
- [ ] Nerf++
- [ ] Neus

#### Polarisation

- [ ] Retrieve polarisation images
- [ ] Depth from polarisation

As the most notebooks contain interactive plots by using plotly, to view it correctly, please paste the notebook link from Github to [**nbviewer**](https://nbviewer.org/).

Please refer the notebooks of different examples I provide. To run them you might need to install following packages.

- pytorch
- pytorch3d
- open3D
- plotly
- opencv

I put colab link in each notebook.

# Pytorch implementation

This repo will focus on implement most of the things using pytorch. It will help make the whole process differentiable. So you have to install pytorch to run this code repo. 
