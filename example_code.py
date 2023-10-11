# Description: This is an example code. It is used to demonstrate how to use the code to train and test the model. Data are generated using the generate_data() function.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

# define fake data generator
def generate_data(): 
    species_list = [i for i in range(110)] # suppose there are total 110 species
    plot_data = []

    # suppose the plot is 25 x 25 = 625 cells.
    for x_axis in range(25):
        for y_axis in range(25):
            x = x_axis  # location 
            y = y_axis  # location 

            individual_number = np.random.randint(40, 110) # generate random number of individuals. The individual number of a cell. 
            individual_in_cell = np.random.choice(species_list, individual_number, replace= True) # generate random species. Assign the species to each individual. 
            true_species_in_cell = len(list(set(individual_in_cell))) # the number of spsecies of a cell.

            rapid_scan_individual = np.random.choice(individual_in_cell, individual_number, replace= True) # subsampling the cell. Simulate the rapid scan. 
            rapid_scan_species = len(list(set(rapid_scan_individual))) # the number of species of a cell-level rapid scan.

            plot_data.append([x, y, rapid_scan_species, true_species_in_cell]) # add the data to the plot_data list.

    # sample cell for training 
    plot_data_idx = [i for i in range(len(plot_data))] # the index of plot_data 
    sampled_idxs = np.random.choice(plot_data_idx, 63, replace= False) # choose the the samples for census. Sample size is about 10% of total cells (625*0.1 = 62.5)

    # separate the plot_data to 10% traing dataset and 90% testing dataset
    training_data = []
    testing_data = []
    for i in range(len(plot_data)):
        if i in sampled_idxs:
            training_data.append(plot_data[i])
        else:
            testing_data.append(plot_data[i])
    return training_data, testing_data

def separate_x_y(data):
    x = []
    y = []
    for i in range(len(data)):
        # add the bias term
        x.append(data[i][0:3] + [1])
        y.append(data[i][3])
    return x, y



# define Mish activation function.
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# initializing data.
train_data, test_data = generate_data()
x, y = separate_x_y(train_data)

# convert data to numpy array. 
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# normalizing data
x_stds = []
x_means = []
y_stds = []
y_means = []
for i in range(len(x[0])):
    mean = np.mean(x[:, i])
    std = np.std(x[:, i])
    x_means.append(mean)
    x_stds.append(std)
x_means[-1] = 0
x_stds[-1] = 1
x_means = np.array(x_means)
x_stds = np.array(x_stds)
y_mean = np.mean(y)
y_std = np.std(y)


x_normalized = []
for i in range(len(x)):
    x_normalized.append((x[i] - x_means)/  x_stds)

y_normalized = []
for i in range(len(y)): 
    y_normalized.append((y[i] - y_mean)/ y_std)

x_normalized = np.array(x_normalized, dtype=np.float32)
y_normalized = np.array(y_normalized, dtype=np.float32)


# define ANN structure 
l1_weight = 1e-4
l2_weight = 1e-4
model = keras.Sequential([
    layers.Input(shape=(4,)),  # input layer 
    layers.Dense(64, activation=mish, kernel_regularizer=regularizers.l1_l2(l1 = l1_weight, l2 = l2_weight)),  # 1st hidden layer 
    layers.Dropout(0.2),  # dropout layer
    layers.Dense(64, activation=mish, kernel_regularizer=regularizers.l1_l2(l1 = l1_weight, l2 = l2_weight)),  # 2nd hidden layer 
    layers.Dropout(0.2),  # dropout layer
    layers.Dense(64, activation=mish, kernel_regularizer=regularizers.l1_l2(l1 = l1_weight, l2 = l2_weight)),  # 3rd hidden layer 
    layers.Dropout(0.2),  # dropout layer
    layers.Dense(64, activation=mish, kernel_regularizer=regularizers.l1_l2(l1 = l1_weight, l2 = l2_weight)),  # 4th hidden layer 
    layers.Dropout(0.2),  # dropout layer
    layers.Dense(64, activation=mish, kernel_regularizer=regularizers.l1_l2(l1 = l1_weight, l2 = l2_weight)),  # 5th hidden layer 
    layers.Dropout(0.2),  # dropout layer
    layers.Dense(1)  # output layer
])

# define lose function 
loss_fn = tf.keras.losses.Huber()

# define optimizer, using Adam optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, 
                                     beta_1=0.9, 
                                     beta_2=0.999, 
                                     epsilon=1e-07, 
                                     amsgrad=False) 

# compile model
model.compile(optimizer=optimizer, loss=loss_fn)

# training. 
epochs = 100 # training epochs 
history = model.fit(x_normalized, y_normalized, epochs=epochs)

# testing data
new_x, new_y = separate_x_y(test_data)
new_x_normalized = []

# normalize the data using the mean and std of training data.
for i in range(len(new_x)):
    new_x_normalized.append((new_x[i] - x_means) / x_stds)
input_data = np.array(new_x_normalized, dtype=np.float32)

# make prediction using trained model 
predicted_value = model.predict(input_data)

# invert the normalized data to original data domain using the mean and std of training data.
invert_y = predicted_value * y_std + y_mean

# summary the result
mab = 0
for i in range(len(new_y)):
    mab += abs(new_y[i] - invert_y[i])
mab = mab / len(new_y)

print("Mean Absolute Bias:", mab)