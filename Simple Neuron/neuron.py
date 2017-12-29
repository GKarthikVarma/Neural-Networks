
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

class DisplayPerceptronLearning:
    """
      The following code contains some GUI code from the program provided by Mr. Farhad Kamangar
      Author: Farhad Kamangar
      Title of the Work: Assignment_00
      """

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight = np.array([1.0,1],dtype=np.float64())
        self.bias = 0
        self.class_1 = np.array(np.random.uniform(-10,10,4))
        self.class_2 = np.array(np.random.uniform(-10,10,4))
        self.line_x = 0
        self.line_y = 0
        self.activation_function = "Symmetric Hard Limit"
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=3, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        #weight_1
        self.input_weight_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight 1",
                                            command=lambda event: self.input_weight_slider_callback())
        self.input_weight_slider.set(self.input_weight[0])
        self.input_weight_slider.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider_callback())
        self.input_weight_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #weight_2
        self.input_weight_slider2 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight 2",
                                            command=lambda event: self.input_weight_slider_callback2())
        self.input_weight_slider2.set(self.input_weight[1])
        self.input_weight_slider2.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider_callback2())
        self.input_weight_slider2.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #bias
        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetric Hard Limit", "Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetric Hard Limit")
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_decision_boundary()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

        #########################################################################
        #  Set up the frame for button(s) [Train and random]
        #########################################################################
        # Train
        self.buttons_frame2 = tk.Frame(self.master)
        self.buttons_frame2.grid(row=1, column=2, sticky = tk.N + tk.S + tk.E + tk.W)
        self.buttons_frame2.rowconfigure(0, weight=1)
        self.buttons_frame2.columnconfigure(0, weight=1, uniform = 'xx')
        self.train = tk.Button(self.buttons_frame2, text="Train", justify="center")
        self.train.grid(row=0, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.train.bind("<Button-1>", lambda event: self.train_neuron())


        # Generate
        self.train = tk.Button(self.buttons_frame2, text="Generate", justify="center")
        self.train.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.train.bind("<Button-1>", lambda event: self.generate_random_weights())

        #########################################################################
        #  Plot the Graph
        #########################################################################
    def display_decision_boundary(self):

        #setup bias and weights never == 0
        self.axes.cla()
        if self.bias == 0:
            bias = 0.00001
        else:
            bias = self.bias

        if self.input_weight[0] == 0:
            input_weight_1 = 0.00001
        else:
            input_weight_1 = self.input_weight[0]

        if self.input_weight[1] == 0:
            input_weight_2 = 0.00001
        else:
            input_weight_2 = self.input_weight[1]

        x = np.linspace(-10,10,256)
        y = (-(input_weight_1 * x + bias) / input_weight_2)
        self.axes.plot(x, y)
        upper_y = self.ymax
        lower_y = self.ymin

        self.axes.fill_between(x,y,upper_y, where= y < upper_y, facecolors="green")
        self.axes.fill_between(x,y,lower_y, where= y > lower_y, facecolors="red")

        self.axes.scatter(self.class_1[0],self.class_1[1],c='orange',marker='^',label='Class 1 = -1')
        self.axes.scatter(self.class_1[2],self.class_1[3],c='orange',marker='^')
        self.axes.scatter(self.class_2[0],self.class_1[1],c='blue',marker='d',label='Class 2 = +1')
        self.axes.scatter(self.class_2[2],self.class_1[3],c='blue',marker='d')
        plt.legend()

        self.axes.xaxis.set_visible(True)
        #plt.legend("Class1", "Class2")
        self.axes.yaxis.set_visible(True)

        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        plt.title(self.activation_function)
        self.canvas.draw()

    def input_weight_slider_callback(self):
        self.input_weight[0] = self.input_weight_slider.get()
        self.display_decision_boundary()

    def input_weight_slider_callback2(self):
        self.input_weight[1] = self.input_weight_slider2.get()
        self.display_decision_boundary()

    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        self.display_decision_boundary()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.display_decision_boundary()

        #########################################################################
        #  Adjust weights so that linear and hyperbolic tangent work
        #########################################################################
    def adjust_weights(self):
        if self.input_weight[1] < 0.0001:
            self.input_weight[1] = 0.0001
        elif self.input_weight[1] > 40:
            self.input_weight[1] = 40

        if self.input_weight[0] < 0.0001:
            self.input_weight[0] = 0.0001
        elif self.input_weight[0] > 40:
            self.input_weight[0] = 40

        if self.bias < 0.0001:
            self.bias = 0.0001
        elif self.bias > 40:
            self.bias = 40

    #########################################################################
    #  Generate random weights
    #########################################################################

    def generate_random_weights(self):
        self.class_1 = np.array(np.random.uniform(-10, 10, 4))
        self.class_2 = np.array(np.random.uniform(-10, 10, 4))
        self.display_decision_boundary()

    #########################################################################
    #  train perceptron
    #########################################################################
    def train_neuron(self):
        if self.activation_function == "Symmetric Hard Limit":
            for i in range(1,6):
                for j in range(1,21):
                    #input1
                    a = (self.input_weight[0] * self.class_1[0] + self.input_weight[1] * self.class_1[1]) + self.bias
                    if a < 0:
                        activation = -1
                    else:
                        activation = 1
                    self.input_weight[0] = self.input_weight[0] + ((1 - activation) * self.class_1[0])
                    self.input_weight[1] = self.input_weight[1] + ((1 - activation) * self.class_1[1])
                    self.bias = self.bias + (1 - activation)

                    # input2
                    a = (self.input_weight[0] * self.class_1[2] + self.input_weight[1] * self.class_1[3]) + self.bias
                    if a < 0:
                        activation = -1
                    else:
                        activation = 1
                    self.input_weight[0] = self.input_weight[0] + ((1 - activation) * self.class_1[2])
                    self.input_weight[1] = self.input_weight[1] + ((1 - activation) * self.class_1[3])
                    self.bias = self.bias + (1 - activation)


                    # input3
                    a = (self.input_weight[0] * self.class_1[0] + self.input_weight[1] * self.class_2[1]) + self.bias
                    if a < 0:
                        activation = -1
                    else:
                        activation = 1
                    self.input_weight[0] = self.input_weight[0] + ((-1 - activation) * self.class_2[0])
                    self.input_weight[1] = self.input_weight[1] + ((-1 - activation) * self.class_2[1])
                    self.bias = self.bias + (-1 - activation)


                    # input4
                    a = (self.input_weight[0] * self.class_1[2] + self.input_weight[1] * self.class_2[3]) + self.bias
                    if a < 0:
                        activation = -1
                    else:
                        activation = 1
                    self.input_weight[0] = self.input_weight[0] + ((-1 - activation) * self.class_2[2])
                    self.input_weight[1] = self.input_weight[1] + ((-1 - activation) * self.class_2[3])
                    self.bias = self.bias + (-1 - activation)
            self.display_decision_boundary()
        elif self.activation_function == "Hyperbolic Tangent":
            for i in range(1,21):
                for j in range(1,6):
                    #input1

                    a = round( (self.input_weight[0] * self.class_1[0] + self.input_weight[1] * self.class_1[1]) + self.bias, 2)
                    activation = round((np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a)),2)
                    self.input_weight[0] = self.input_weight[0] + ((1 - activation) * self.class_1[0])
                    self.input_weight[1] = self.input_weight[1] + ((1 - activation) * self.class_1[1])
                    self.bias = self.bias + (1 - activation)

                    # input2
                    a = round((self.input_weight[0] * self.class_1[2] + self.input_weight[1] * self.class_1[3]) + self.bias, 2)
                    activation = round((np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a)),2)
                    self.input_weight[0] = self.input_weight[0] + ((1 - activation) * self.class_1[2])
                    self.adjust_weights()

                    self.input_weight[1] = self.input_weight[1] + ((1 - activation) * self.class_1[3])
                    self.adjust_weights()

                    self.bias = self.bias + (1 - activation)
                    # input3
                    a = round((self.input_weight[0] * self.class_1[0] + self.input_weight[1] * self.class_2[1]) + self.bias,2)
                    activation = round((np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a)),2)
                    self.input_weight[0] = self.input_weight[0] + ((-1 - activation) * self.class_2[0])
                    self.input_weight[1] = self.input_weight[1] + ((-1 - activation) * self.class_2[1])
                    self.bias = self.bias + (-1 - activation)
                    self.adjust_weights()

                    # input4
                    a = round((self.input_weight[0] * self.class_1[2] + self.input_weight[1] * self.class_2[3]) + self.bias,2)
                    activation = round((np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a)),2)
                    self.input_weight[0] = self.input_weight[0] + ((-1 - activation) * self.class_2[2])
                    self.input_weight[1] = self.input_weight[1] + ((-1 - activation) * self.class_2[3])
                    self.bias = self.bias + (-1 - activation)
                    self.adjust_weights()
            self.display_decision_boundary()
        elif self.activation_function == "Linear":
            for i in range(1,21):
                for j in range(1,6):
                    #input1
                    a = (self.input_weight[0] * self.class_1[0] + self.input_weight[1] * self.class_1[1]) + self.bias
                    activation = a
                    self.input_weight[0] = self.input_weight[0] + ((1 - activation) * self.class_1[0])
                    self.input_weight[1] = self.input_weight[1] + ((1 - activation) * self.class_1[1])
                    self.bias = self.bias + (1 - activation)
                    self.adjust_weights()

                    # input2
                    a = (self.input_weight[0] * self.class_1[2] + self.input_weight[1] * self.class_1[3]) + self.bias
                    activation = a
                    self.input_weight[0] = self.input_weight[0] + ((1 - activation) * self.class_1[2])
                    self.input_weight[1] = self.input_weight[1] + ((1 - activation) * self.class_1[3])
                    self.bias = self.bias + (1 - activation)
                    self.adjust_weights()

                    # input3
                    a = (self.input_weight[0] * self.class_1[0] + self.input_weight[1] * self.class_2[1]) + self.bias
                    activation = a
                    self.input_weight[0] = self.input_weight[0] + ((-1 - activation) * self.class_2[0])
                    self.input_weight[1] = self.input_weight[1] + ((-1 - activation) * self.class_2[1])
                    self.bias = self.bias + (-1 - activation)
                    self.adjust_weights()

                    # input4
                    a = (self.input_weight[0] * self.class_1[2] + self.input_weight[1] * self.class_2[3]) + self.bias
                    activation = a
                    self.input_weight[0] = self.input_weight[0] + ((-1 - activation) * self.class_2[2])
                    self.input_weight[1] = self.input_weight[1] + ((-1 - activation) * self.class_2[3])
                    self.bias = self.bias + (-1 - activation)
                    self.adjust_weights()
            self.display_decision_boundary()

root = tk.Tk()
root.title("Assignment 01 -- Gadiraju")
window = DisplayPerceptronLearning(root, root)
root.mainloop()
