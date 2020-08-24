import matplotlib.pyplot as plt
from planar_utils import load_extra_datasets, load_planar_dataset, plot_decision_boundary
from NN_Model import Model

X, Y = load_planar_dataset()

# Using all examples for training
n_x = X.shape[0]
n_h = 20                                     # Using 15 hidden Layers
n_y = Y.shape[0]

NN = Model(n_x, n_h, n_y)

NN.fit(X, Y, num_iterations = 10000, print_cost = True)

predictions = NN.predict(X)

accuracy = NN.accuracy(X, Y)

print(f"Accuracy = {accuracy}%")

plot_decision_boundary(lambda x: NN.predict(x.T), X, Y)
plt.title("Decision boundary for hidden layer" + str(20))

plt.show()
