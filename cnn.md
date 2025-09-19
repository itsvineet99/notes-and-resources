## **chapter 14:** Classifying Images with Deep Convolutional Neural Networks

### temp/

- cnn arch : convolution blocks ($Z=W*X+b$) -> activation layers ($A=\sigma(Z)$), pooling layers to 
- The capacity of a network refers to the level of complexity of the function that it can learn to approximate.

### Understanding CNNs and feature hierarchies

- CNN's were inspired by how visual cortex of human brain works.
- their exists layers of visual cortex, primary layers -> edges, straight lines, simple features, higher order-layers -> complex shapes and features.
- Yann LeCun and his colleagues invented CNN in 1989 and then in 2019 he received turing award.
- convolutional architecture -> feature extraction layers 
- in CNN, the early (layers that are right after input layer) extract low-level features from raw data and the later layers (fully connected mlp layers) use these features to make prediction 
- cnn extracts low-level feature and then combines them to build high level features, like in case of images they will extract features like edges, blobs, straight lines and then combine them to build to complex shapes like buildings, cats or dogs. this is how we construct so called **feature hierarchy**.

- in CNN we create a **feature map** from input image, where each element of feature map comes from local patch of pixels in the input image. this local path is also called **local receptive field**.

![feature map diagram](https://i.ibb.co/23ZCrhk8/Screenshot-2025-09-14-at-12-11-22-PM.png)


why cnn's perform better on images?

- **sparce connectivity**: as we can see only a few pixels are connected to a single feature element (unlike mlp where entire image is connected to single feature i.e layers are fully connected)
- **parameter sharing**: same weights are used for different patches of input image. which benefits us by substantially decreasing the number of weights used in the network 
- **translation invariance**: dnn's learn to detect a specific object at specific position only and hence if same object that it has learned to recognize on left corner it won't be able to to recognize it if the object appears in right corner. but cnn's on the other hand, due to their shared weights of kernel can regonize same object at any position of image.

- CNN -> convolutional layers + subsampling layers (pooling layers) + fully connected layer (mlp)
- pooling layers don't have biases and weights but convolutional layer and fully connected layers do.

### discrete convolutional in one dimension 

`*` here stands as a convolutional operation and not as a multiplication operation.

convolution between two vectors x and w is denoted as $y=x*w$, where $x$ is called input and $w$ is called **filter** or **kernel**.

mathematical operation inside a convolution looks like given formula,

$$
y = x * w \;\;\Rightarrow\;\; y[i] = \sum_{k=-\infty}^{+\infty} x[i-k] \, w[k]
$$

here [] are used to show indexing inside each vector. also given above is a theoretical formula used in signal processing, we don't use this exact formula in deep learning or any computation as we have finite inputs unlike infinite signals that are we assuming here.

so what's actually happening in above case we are calculating the $ith$ output $y[i]$ by doing weighted addition of elements of input patch and weights (kernel elements) but by flipping the weights i.e 1st input multiplied with last output, second with second last etc. we can also understand it like we are doing dot product between input patch and weights (but flipped).

for computation purposes in real life we use given formula,

$$
\mathbf{y} = \mathbf{x} * \mathbf{w} \;\;\Rightarrow\;\;
y[i] = \sum_{k=0}^{m-1} x^{p}[i+m-k] \, w[k]
$$

here $m$ is the size of of weight vector i.e kernel have $m$ elements in it. and $x^p$ is the padded input we'll talk about padding in next section.

so what's happening in above formula? for each $1st$ output, we are going from 0 to $m-1$th element in kernel, and for input $x$ we are going from $mth$ element of $x$ to $1st$. for $2nd$ output kernel is iterated in same fashion but for input $x$ we start from $m+1$ and go till $2$ and this is how we slide the window one step to the right and repeat the calculation, until all the elements in input are iterated over.

**concrete example on above formula**:

let's take a concrete example and see how we do convolution on example where $m=4$, kernel have 4 elements

 **Exact Calculation for the 1st Output (y[0])**

Here, `i=0` and `m=4`. The index for the input `x` becomes `x[0 + 4 - k]` or `x[4-k]`. We sum from `k=0` to `3`.

- When `k=0`, we take `x[4-0] * w[0]` --> `x[4] * w[0]`
    
- When `k=1`, we take `x[4-1] * w[1]` --> `x[3] * w[1]`
    
- When `k=2`, we take `x[4-2] * w[2]` --> `x[2] * w[2]`
    
- When `k=3`, we take `x[4-3] * w[3]` --> `x[1] * w[3]`
    

So, the exact calculation for `y[0]` is: `y[0] = x[1]w[3] + x[2]w[2] + x[3]w[1] + x[4]w[0]`

This operation takes the input patch from `x[1]` to `x[4]` and multiplies it with the kernel `w` where the kernel is effectively "flipped" relative to the input.

---

**Exact Calculation for the 2nd Output (y[1])**

Now, `i=1` and `m=4`. The index for the input `x` becomes `x[1 + 4 - k]` or `x[5-k]`. We sum from `k=0` to `3`.

- When `k=0`, we take `x[5-0] * w[0]` --> `x[5] * w[0]`
    
- When `k=1`, we take `x[5-1] * w[1]` --> `x[4] * w[1]`
    
- When `k=2`, we take `x[5-2] * w[2]` --> `x[3] * w[2]`
    
- When `k=3`, we take `x[5-3] * w[3]` --> `x[2] * w[3]`
    

So, the exact calculation for `y[1]` is: `y[1] = x[2]w[3] + x[3]w[2] + x[4]w[1] + x[5]w[0]`

Notice that the input window has simply **slid one position to the right** (it now uses `x[2]` to `x[5]`), and the same flipped multiplication process is repeated.


> so in short this is what we do in convolution, we flip the input and then do dot product of input patch (not the entire input feature set but just the patch which is of size of kernel) with kernel and then we keep sliding this patch to the right slide for each output.

> convolution is  commutative in nature so it doesn't matter if we flip the input or if we flip the kernel, so for computation purpose and easy to understand we flip the kernel and then do sliding dot product which is also given in the image bellow

let's visually look how convolution looks,

![convolution](https://i.ibb.co/1xsTy4w/Screenshot-2025-09-14-at-7-52-26-PM.png)


this is all good but deep learning libraries like pytorch and tensorflow don't really implement convolution this exact same way. the do **cross-correlation** and call it convolution.

this is the formula of cross-correlation,

$$
y = x \star w \;\;\Rightarrow\;\; y[i] = \sum_{k=-\infty}^{+\infty} x[i+k] \, w[k]
$$

cross-correlation is same as convolution, like in this also we do sliding dot product between input patch and kernel but the difference here is we don't flip the kernel or input, dot product happens here in same direction and this is the main difference between convolution and cross-correlation.

why do libraries use cross-correlation instead of convolution?

- for deep learning network (CNN) it only wants to learn the representation or features of image, and it does so by updating its weights accordingly so the only goal here is to learn the weights of kernel. hence it doesn't matter if kernel is flipped or not. all that matters is we learn the weights of kernel
- another reason is flipping the kernel adds an additional step, which adds additional computation, and doing it for millions of parameters increases computational cost by large margine so we save computational cost by not flipping the kernel.

dl libraries implement cross-correlation and still call it convolutional just because of its convention.

### padding

padding is adding more elements to the both sides of original inputs. in case of images it could be understood as adding more pixels on boundries.

**why we do padding?**

- when we perform convolution on input with kernel, the output we get has smaller size then original input. and in neural networks it can cause problem as we have many layers so with each convolution layer our outputs will keep on shrinking and in that process we will lose lot of important information.
- another reason for adding padding is, the pixels on boundaries are involved in less number of computations then pixels in middle, due to which we don't pay much attention to pixels on boundaries and can lose important information from those pixels

**what do i mean by not paying attention on pixels on boundaries?**

Now, consider an example where $n = 5$ and $m = 3$. Then, with $p = 0$, x[0] is only used in computing one output element (for instance, y[0]), while x[1] is used in the computation of two output elements (for instance, y[0] and y[1]).

in more simpler words first we take a patch of input x, this patch starts from x[0] till m-1 and we use it to compute y[0], then we slide our kernel in right side and now our patch starts from x[1] and we use this input to get y[1], from this we can see that we only used x[0] once and x[1] twice and thats how we pixels at boundaries get less attention.

**types of padding**

padding has three types:
- full padding: 
	- generates greater output than input
	- padding size is calculated by $p=m-1$
	- not used in cnn as we don't need bigger output then input
- same padding:
	- output == input
	-  the padding parameter, p, is computed according to the filter size, along with the requirement that the input size and output size are the same.
	- this is the one used in cnn, as we want our input and output size same
- valid padding or no padding
	- output is smaller than input
	- no padding is applied here, convolution happens directly on input
	- not used in cnn for obvious reasons stated above

![padding types](https://i.ibb.co/sdTsS7tD/Screenshot-2025-09-15-at-3-09-54-PM.png)


### determining the size of convolutional output

for the convolution operation $y=x*w$, where $x$ is of size $n$ and w is of size $m$ and padding size is $p$ and stride is $s$ (stride is the size of steps we skip while shifting).

formula for output looks like this,

$$
0=|\frac{n+2p-m}{s}|+1
$$

here $||$ is a flooring operation.

### 2D convolution

all the concepts that we learned in 1D convolution also applies to 2D convolution, only difference is we add one more dimension.

assuming we have 2 dimensional input and kernel, where $m_{1}\times m_{2}$ is the size of kernel and $n_{1}\times n_{2}$ is the size of input image, given will be the formula (theoretical formula),

$$
\mathbf{Y} = \mathbf{X} * \mathbf{W} \rightarrow Y[i,j] = \sum_{k_1=-\infty}^{+\infty} \sum_{k_2=-\infty}^{+\infty} X[i-k_1, j-k_2] W[k_1, k_2]
$$

and given is the cross-correlation formula that libraries like pytorch uses,

$$
Y[i,j] = \sum_{k_1=0}^{m_1-1} \sum_{k_2=0}^{m_2-1} X[i+k_1, j+k_2] W[k_1, k_2]
$$


visual representation of how 2D convolution works, here we are using $3\times 3$ kernel and $8\times 8$ image to create $8\times8$ output, 

![2D conv.png](https://i.ibb.co/QjHZfqLr/Screenshot-2025-09-15-at-5-01-03-PM.png)

let's look at the example to understand it more deeply,

![example.png](https://i.ibb.co/Jw5nwHYb/Screenshot-2025-09-15-at-5-08-07-PM.png)
![example.png](https://i.ibb.co/5gQdGkwL/Screenshot-2025-09-15-at-5-09-48-PM.png)

**code implementation:**

```python 
import numpy as np
import scipy.signal

def conv2D(x, w, p=(0,0), s=(1,1)):
	w_rot = np.array(w)[::-1, ::-1] # rotating kernel
	x_orig = np.array(x)
	
	# creating new dimensions for padded input
	n_1 = x_orig.shape[0] + 2*p[0]
	n_2 = x_orig.shape[1] + 2*p[1]
	
	# padding input fr
	x_padded = np.zeros(shape=(n_1, n_2))
	x_padded[p[0]: p[0]+x_orig.shape[0],
			 p[1]: p[1]+x_orig.shape[1]] = x_orig
			 
	# let's write logic for convolution, here first loop iterates over each row
	# and second loop iterates over each column, loop starts from 0 and go till
	# size of output and stride is the step we skip.
	res = []
	for i in range(0, int((x_padded.shape[0]-w_rot.shape[0])/s[0])+1, s[0]):
		res.append([])
		for j in range(0, int((x_padded.shape[1]-w_rot.shape[1])/s[1])+1, s[1]):
			x_sub = x_padded[i: i+w_rot.shape[0],
							 j: j+w_rot.shape[1]]
			res[-1].append(np.sum(x_sub * w_rot)) # -1 index as we are adding these in last list in list.
	return (np.array(res))
  
# lets take a example

X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]

print(f'conv2D implementation: \n {conv2D(X, W, p=(1,1), s=(1,1))}')
print(f'scipy results: \n {scipy.signal.convolve2d(X, W, mode='same')}')
```


### subsampling layers 

![pooling.png](https://i.ibb.co/mVnRxWgP/Screenshot-2025-09-15-at-6-28-08-PM.png)

so pooling is what we are seeing the above image, we fix the size of neighbourhood such as $P_{3\times 3}$, this size is also called as pooling size.

then depending upon type of pooling (mean or max) we perform the pooling operation.

in **max pooling** we take the largest element from neighbourhood, and in **mean pooling** we take the average of all the elements in neighbourhood.

**advantages of pooling**:

 pooling (max pooling) introduces local invariance i.e we focus on general features of input and can avoid noise or our output is robust and we don't focus on small shifts or translation in local region. local invariance causes us to ignore small shifts in local region and focus on the dominant characteristics.
 
for example look at the given example where we have two different input patches but they r mostly same with lil bit of shifting and noise,

$$
\mathbf{X}_2 = 
\begin{bmatrix} 
100 & 100 & 100 & 50 & 100 & 50 \\
95 & 255 & 100 & 125 & 125 & 170 \\
80 & 40 & 10 & 10 & 125 & 150 \\
255 & 30 & 150 & 20 & 120 & 125 \\
30 & 30 & 150 & 100 & 70 & 70 \\
70 & 30 & 100 & 200 & 70 & 95 
\end{bmatrix}

\xrightarrow{\text{max pooling } P_{2\times2}}

\begin{bmatrix} 
255 & 125 & 170 \\
255 & 150 & 150 \\
70 & 200 & 95 
\end{bmatrix}
$$

$$
\mathbf{X}_1 = 
\begin{bmatrix} 
10 & 255 & 125 & 0 & 170 & 100 \\
70 & 255 & 105 & 25 & 25 & 70 \\
255 & 0 & 150 & 0 & 10 & 10 \\
0 & 255 & 10 & 10 & 150 & 20 \\
70 & 15 & 200 & 100 & 95 & 0 \\
35 & 25 & 100 & 20 & 0 & 60 
\end{bmatrix}
\quad

\xrightarrow{\text{max pooling } P_{2\times2}}

\begin{bmatrix} 
255 & 125 & 170 \\
255 & 150 & 150 \\
70 & 200 & 95 
\end{bmatrix}
$$

we can see that after performing max pooling we get the same answer, i.e *we achieved local invariance and our output is robust to local shifts*

another reason pooling is effective is because it decreases the size of features, which increases the computational effectiveness.

Traditionally, pooling is assumed to be non-overlapping. Pooling is typically performed on non-overlapping neighbourhoods, which can be done by setting the stride parameter equal to the pooling size. For example, a non-overlapping pooling layer, $P_{n_{1}\times n_{2}}$ , requires a stride parameter $S=(n_{1}, n_{2})$.

we can also have CNN without pooling layers, in such a architecture we add convolutional layers with stride 2 as a pooling layer to reduce the features size.


### dealing with multiple inputs in CNN

conventional way to give input to convolution layers is three dimensional like $X_{N_{1}\times N_{2}\times C_{in}}$, where $C_{in}$ is the number of input channels.

for a first layer in cnn, we have image as a input and if image is in RGB color mode then $C_{in}=3$ (for the red, green, and blue color channels in RGB)and if image is grey scale then $C_{in}=1$.

>Note that with torchvision, the input and output image tensors are in the format of Tensor[channels, image_height, image_width].

each channel represents different color and when we select only one channel value while slicing then we get the values of intensities of that color.

**given are the formulae for how we perform convolutions in CNN,**

![formula.png](https://i.ibb.co/kVMyRNTQ/Screenshot-2025-09-17-at-7-02-44-PM.png)

- this image that we got above is for when we generate only one output with our convolution layer and hence we have only one filter or say weight matrix (weight matrix == filter / kernel).
- another thing to notice is that we are performing convolution for each channel input, yes we do that in CNN, we perform convolution for each channel input. and this is the reason we have different filter for each channel input.
- so here's what happens in convolution layer in CNN, we perform convolution between input and filter for each input channel separately, and then we add the outputs we get. after addition we get pre-activation value $Z$. we add bias in this value and pass it to activation layer, and after performing activation function on it we get output $A$.


![formula2.png](https://i.ibb.co/Ld9vdNkK/Screenshot-2025-09-17-at-7-11-31-PM.png)

- what separates above formula from previous formula is now our convolution layer is not just producing one output but we are producing multiple outputs and hence our weight matrix is now 4 dimensional i.d $W_{m_{1}\times m_{2}\times C_{in}\times C_{out}}$.
- now here each operation happens separately for each output.

> one of the intuition about cnn's is here also we have weights but instead of one weight (like in vanilla neural network) each weight is multidimensional and its an 2D array or matrix.

>filter is to CNN what a single weight is to an MLP, just like each weight in mlp maps single feature to a neuron the fileter maps an entire image (but for one channel input only) to a neuron i.e in a way filter is just like a single in mlp but for cnn


to understand how less parameters cnn uses let's look at one example:

for a RGB image with dimension (height * width) $m_{1}\times m_{2}$, and if we assume that we need 5 outputs, the no. of parameters will look like this $m_{1}\times m_{2}\times 3\times 5$  which is exactly equal to the weight matrix (lol this what happens no. parameters == no. weights cause weights are parameters lol) for 5 outputs we will also have 5 biases so the total will become 

$$
m_{1}\times m_{2}\times 3\times 5+5
$$

the output that we will get after this will be of size $m_{1}\times m_{2}\times 3$.

now let's assume same scenario for normal neural network with input size $n_{1}\times n_{2}$, as we know for NN no. of weights are calculated by input size * output size hence, number of parameters in normal NN will be, 

$$
(n_{1}\times n_{2} \times 3)(n_{1}\times n_{2}\times 5)=(n_{1}\times n_{2})^2\times 3 \times 5
$$

in addition there will bias vector of size $(n_{1}\times n_{2}\times 5)$, also the size of kernel is way smaller then the size of images i.e $m_{1}<n_{1}$  and $m_{2}<n_{2}$ and now we can see the difference between trainable parameters in cnn and in vanilla neural network.

### L2 regularization and dropout

- L2 regularization is just adding regularization term in a loss to increase the loss, so that our model don't overfit
- to calculate this regularization term we have a formula where we add the square of all the weights and then multiply it with hyper parameter lambda.

**Dropout**:

- dropout is a method where we randomly drop weights during each training iteration.
- we set the probability of droping out for each weight to be $P_{drop}$ which is generally $0.5$
- we also have $P_{\text{keep}}$ which is probability to keep weights i.e $1-P_{drop}$
- dropping out weights randomly makes our model generalize better and although the reasons are not clear why that is but one of the intuition is it acts like a ensembling.
- as we know ensembled models perform better than individual models, when we randomly dropout weights in each iteration, we are basically training new model for each iteration and then during prediction we don't drop any weights which means we are averaging over all these models so our predictions becomes better due to dropout.
- usually the neurons that get deactivated or weights that are dropped are in hidden layers
- as drop out randomly drops the neurons, the neurons from next layers can't be dependent on some specific neuron for specific type of information (which is reason for overfitting) so it learns to generalize and be balanced so it can take information from any layers and be able to make good predictions
- so have $2^h$ different combinations that model can exist in (as we set p=0.5, it makes this situation binary combination type thing i.e 2 * 2 h times).
- now as we know dropout is like ensembling, but in ensemble models for prediction we run our output through all the models and then take the average of their predictions but this is not possible in neural networks as it get very expensive computationally hence we have a mathematical shortcut in dropout which gives us the same results without using that much computation i.e by scaling the prediction.
- we first make prediction with our trained model then after making the predictions we scale those prediction by a specific factor this factor is $\frac{1}{1-P_{drop}}$


### loss functions for classification

- binary cross entropy -> binary classification
- categorical cross-entropy -> multiclass classification
-  note that computing the cross-entropy loss by providing the logits, and not the class-membership probabilities, is usually preferred due to numerical stability reasons.

### CNN architecture

- input dimensions -> $28\times 28\times 1$ (cause of grayscale image input channel is 1)
- convolutional layer 1 -> kernel size : $5\times 5$ -> 32 outputs 
- pooling layer 1 -> $P_{2\times 2}$
- convolutional layer 2 -> kernel size : $5\times 5$ -> 64 outputs
- pooling layer 1 -> $P_{2\times 2}$ 
- flattened 
- fully connected layer -> outputs 1024
- fully connected layer with softmax to make prediction 

![CNN.png](https://i.ibb.co/mFDP1TsC/Screenshot-2025-09-18-at-5-35-09-PM.png)

The dimensions of the tensors in each layer are as follows:

* Input: `[batchsize×28×28×1]`
* Conv_1: `[batchsize×28×28×32]`
* Pooling_1: `[batchsize×14×14×32]`
* Conv_2: `[batchsize×14×14×64]`
* Pooling_2: `[batchsize×7×7×64]`
* FC_1: `[batchsize×1024]`
* FC_2 and softmax layer: `[batchsize×10]`

- for convolution layers, stride = 1.
- for pooling layers, stride = 2.

- **NCHW format** -> no. of images, no. of channels, height, width


### data augmentation

- its a technique we use when we have limited training data
- here we increase the size of training data by using different methods to duplicate the data like synthesizing new data, modifying data.
- in case of images we can augment data with methods like,  cropping parts of an image, flipping, and changing the contrast, brightness, and saturation.
- we keep randomization in our augmentation, like we don't change contrast of all the images or on the same level, we change the contrast randomly and with random values.