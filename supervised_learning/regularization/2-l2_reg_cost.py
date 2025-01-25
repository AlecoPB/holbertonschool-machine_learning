#!/usr/bin/env python3
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Load the model
model = load_model('model_reg.h5')

# Assume `cost` is the unregularized cost (cross-entropy loss) calculated elsewhere
cost = K.variable(0.5)  # Example tensor value for base cost

# Calculate the total cost with L2 regularization
total_cost = l2_reg_cost(cost, model)

# Evaluate the result (e.g., during training)
print("Total Cost with L2 Regularization:", K.eval(total_cost))
