# Program to train a neuron to distinguish 2 linearly seperable classes.

### Sliders
+ Slider 1: Change w1 (first weight) between -10 and 10. Default value = 1
+ Slider 2: Change w2 (second weight) between -10 and 10. Default value = 1
+ Slider 3: Change b (bias between -10 and 10. Default value=0
 
### Buttons
+ Button1: Train. Clicking this button should adjust the weights and bias for 100 steps using the learning rule. Wnew =Wold+ epT  where  e = t – a
+ Button 2: Create random data. Assuming that there are only two possible target values 1 and -1 (two classes), this button should create 4 random data points (two points for each class). The range of data points should be from -10 to 10 for both dimensions.
 
### Drop Down Selection
The drop down box should allow the user to select between three transfer functions (Symmetrical Hard limit, Hyperbolic Tangent, and Linear)
