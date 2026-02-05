# Neural Network Architecture Search (dataGenerator.c)

A dynamic neural network implementation that systematically searches for optimal architectures on the MNIST dataset by varying the number of hidden layers (0-10) and nodes per layer (1-1000).

## Overview

This program benchmarks different neural network configurations to find which architecture achieves 100% accuracy fastest. For each configuration, it runs 10 training sessions and reports average epochs and time required.

## Algorithm

### Architecture Progression

```
Hidden Layers: 0 → 1 → 2 → ... → 10
Nodes per layer: 1 → 2 → ... → 128 (Carry-based increment)
```

**Example progression for 2 hidden layers:**
```
784 : 1 : 1 : 10
784 : 1 : 2 : 10
...
784 : 1 : 128 : 10
784 : 2 : 1 : 10
...
```

The algorithm increments the nodes in the first hidden layer, carrying over to the next layer when `NODES_CAP` is reached, similar to a number system.

The algorithm always increments the layer with the **fewest nodes** first, ensuring balanced architectures.

---

## Neural Network Implementation

### 1. Network Structure

```c
typedef struct {
    int num_layers;       // Total layers including input/output
    int* layer_sizes;     // Size of each layer
    float** weights;      // weights[i] = connections from layer i to i+1
    float** biases;       // biases[i] = biases for layer i+1
    float** outputs;      // outputs[i] = activations of layer i
    float** errors;       // errors[i] = gradients at layer i
} NeuralNetwork;
```

### 2. Weight Initialization (He Initialization)

For each weight connecting layer with `n` input nodes:

```
weight = random(-1, 1) × √(2/n)
```

**Why He initialization?** It prevents vanishing/exploding gradients with ReLU activation.

### 3. Forward Propagation

For each layer `l` from input to output:

```
z[l+1] = weights[l] × outputs[l] + biases[l]
outputs[l+1] = activation(z[l+1])
```

**Activation functions:**
- Hidden layers: **ReLU** → `max(0, x)`
- Output layer: **Softmax** → `exp(x_i) / Σexp(x_j)`

**Numerically stable softmax:**
```c
max_val = max(output)
exp_sum = Σ exp(output[i] - max_val)
softmax[i] = exp(output[i] - max_val) / exp_sum
```

### 4. Backward Propagation

**Output layer error (Cross-Entropy + Softmax):**
```
error_output[i] = output[i] - target[i]
```
Where `target` is one-hot encoded (1 for correct class, 0 otherwise).

**Hidden layer errors:**
```
error[l] = (weights[l]ᵀ × error[l+1]) ⊙ ReLU'(outputs[l])
```
Where `ReLU'(x) = 1 if x > 0, else 0`.

**Weight updates (SGD):**
```
weights[l] -= learning_rate × error[l+1] × outputs[l]ᵀ
biases[l] -= learning_rate × error[l+1]
```

---

## Training Loop

```
FOR each architecture configuration:
    FOR run = 1 to 10:
        Initialize network with random weights
        Start timer
        
        FOR epoch = 1 to EPOCHS_CAP (100):
            # Training phase
            FOR each training image:
                Forward pass
                Backward pass (update weights)
            
            # Testing phase
            accuracy = test on 10,000 images
            
            IF accuracy >= 100%:
                BREAK
        
        Record time and epochs
    
    Print averages
```

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `INPUT_NODES` | 784 | 28×28 pixel images |
| `OUTPUT_NODES` | 10 | Digits 0-9 |
| `LEARNING_RATE` | 0.01 | SGD step size |
| `MAX_LAYERS` | 12 | Input + 10 hidden + Output |
| `NODES_CAP` | 128 | Max nodes per hidden layer |
| `EPOCHS_CAP` | 20 | Max epochs per run |
| `RUNS_PER_CONFIG` | 5 | Runs averaged per architecture |
| `TARGET_ACCURACY` | 0.98 | 98% accuracy target |

---

## Output Format

For each architecture tested:

```
========================================
Testing: 2 hidden layers
Architecture: 784 : 128 : 64 : 10
========================================
  Run 1: Epochs=15, Time=1.34s, Achieved100=YES
  Run 2: Epochs=18, Time=1.51s, Achieved100=YES
  ...
  Run 5: Epochs=20, Time=1.87s, Achieved100=NO

------ RESULTS ------
TOTAL EPOCHS AVERAGE : 17.30
ARCHITECTURE : 784 : 128 : 64 : 10
TOTAL TIME AVERAGE : 1.45 sec
98.00% ACHIEVED : 4/5 runs
```

### CSV Logging

The program automatically logs results to CSV files in the execution directory:

- **`runs.csv`**: Detailed logs for every individual training run.
- **`results.csv`**: Aggregated results for each architecture configuration.

---

## Compilation & Execution

```bash
# Compile with optimizations
gcc -O3 -march=native -ffast-math dataGenerator.c -o dataGenerator -lm

# Run
./dataGenerator
```

**Optimization flags:**
- `-O3`: Maximum optimization level
- `-march=native`: CPU-specific optimizations
- `-ffast-math`: Faster floating-point (less precise)
- `-lm`: Link math library

---

## File Structure

```
Number-Detector/
├── MNIST/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── Research/
    ├── dataGenerator.c
    └── README.md
```

> ⚠️ **Note:** Update the `TRAIN_IMAGES_PATH`, etc. defines in `dataGenerator.c` if your MNIST files are in a different location.

---

## Performance Considerations

1. **Memory**: Large networks (1000 nodes × 10 layers) require significant RAM
2. **Time**: Full search takes hours; node steps (1→10→100→1000) reduce iterations
3. **Accuracy**: 100% on MNIST is extremely rare; most configs will hit EPOCHS_CAP

---

## Future Improvements

- [ ] Add batch training for faster convergence
- [ ] Implement Adam optimizer instead of SGD
- [ ] Add early stopping based on validation loss
- [ ] Parallelize runs with OpenMP/pthreads
