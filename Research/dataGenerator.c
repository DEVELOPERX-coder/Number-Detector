#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>

#define TRAIN_IMAGES_PATH "../MNIST/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "../MNIST/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "../MNIST/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "../MNIST/t10k-labels.idx1-ubyte"

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define TRAIN_IMAGE_COUNT 60000
#define TEST_IMAGE_COUNT 10000

#define INPUT_NODES 784
#define OUTPUT_NODES 10
#define LEARNING_RATE 0.01f

#define MAX_LAYERS 12        // Input + max 10 hidden + Output
#define NODES_CAP 128
#define EPOCHS_CAP 20
#define RUNS_PER_CONFIG 5
#define TARGET_ACCURACY 0.98f // 98%

// Neural Network structure with dynamic layers
typedef struct {
    int num_layers;
    int* layer_sizes;           // Size of each layer
    float** weights;            // weights[i] = weights from layer i to i+1
    float** biases;             // biases[i] = biases for layer i+1
    float** outputs;            // outputs[i] = output of layer i
    float** errors;             // errors[i] = error at layer i
} NeuralNetwork;

// ======================== MNIST READING ========================

uint32_t swap_endian(uint32_t value) {
    return ((value << 24) & 0xFF000000) |
           ((value << 8)  & 0x00FF0000) |
           ((value >> 8)  & 0x0000FF00) |
           ((value >> 24) & 0x000000FF);
}

bool read_MNIST_images(const char* filename, float** pixels_float) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return false;
    }

    uint32_t magic_number, num_of_images, num_of_rows, num_of_cols;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = swap_endian(magic_number);
    if (magic_number != 2051) {
        fprintf(stderr, "Error: Invalid Magic Number %u (expected 2051)\n", magic_number);
        fclose(file);
        return false;
    }

    fread(&num_of_images, sizeof(num_of_images), 1, file);
    num_of_images = swap_endian(num_of_images);

    fread(&num_of_rows, sizeof(num_of_rows), 1, file);
    num_of_rows = swap_endian(num_of_rows);

    fread(&num_of_cols, sizeof(num_of_cols), 1, file);
    num_of_cols = swap_endian(num_of_cols);

    if (num_of_rows != IMAGE_ROWS || num_of_cols != IMAGE_COLS) {
        fprintf(stderr, "Image dimension mismatch in %s\n", filename);
        fclose(file);
        return false;
    }

    size_t total_size = (size_t)num_of_images * num_of_rows * num_of_cols;
    uint8_t* pixels = malloc(total_size);
    if (pixels == NULL) {
        fprintf(stderr, "Error: Pixels Memory Allocation Failed\n");
        fclose(file);
        return false;
    }

    fread(pixels, 1, total_size, file);
    fclose(file);

    *pixels_float = malloc(total_size * sizeof(float));
    if (*pixels_float == NULL) {
        free(pixels);
        fprintf(stderr, "Error: Pixels_Float Memory Allocation Failed\n");
        return false;
    }

    for (size_t i = 0; i < total_size; ++i) {
        (*pixels_float)[i] = (float)pixels[i] / 255.0f;
    }

    free(pixels);
    return true;
}

bool read_MNIST_labels(const char* filename, uint8_t** labels) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return false;
    }

    uint32_t magic_number, num_of_labels;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = swap_endian(magic_number);
    if (magic_number != 2049) {
        fprintf(stderr, "Error: Invalid Magic Number %u (expected 2049)\n", magic_number);
        fclose(file);
        return false;
    }

    fread(&num_of_labels, sizeof(num_of_labels), 1, file);
    num_of_labels = swap_endian(num_of_labels);

    *labels = malloc(num_of_labels);
    if (*labels == NULL) {
        fprintf(stderr, "Error: Labels Memory Allocation Failed\n");
        fclose(file);
        return false;
    }
    fread(*labels, 1, num_of_labels, file);

    fclose(file);
    return true;
}

// ======================== NEURAL NETWORK ========================

NeuralNetwork* create_network(int* layer_sizes, int num_layers) {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if (!nn) return NULL;

    nn->num_layers = num_layers;
    nn->layer_sizes = malloc(num_layers * sizeof(int));
    nn->weights = malloc((num_layers - 1) * sizeof(float*));
    nn->biases = malloc((num_layers - 1) * sizeof(float*));
    nn->outputs = malloc(num_layers * sizeof(float*));
    nn->errors = malloc(num_layers * sizeof(float*));

    if (!nn->layer_sizes || !nn->weights || !nn->biases || !nn->outputs || !nn->errors) {
        free(nn);
        return NULL;
    }

    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    for (int i = 0; i < num_layers; ++i) {
        nn->outputs[i] = malloc(layer_sizes[i] * sizeof(float));
        nn->errors[i] = malloc(layer_sizes[i] * sizeof(float));
        if (!nn->outputs[i] || !nn->errors[i]) return NULL;
    }

    for (int i = 0; i < num_layers - 1; ++i) {
        size_t weight_count = (size_t)layer_sizes[i] * layer_sizes[i + 1];
        nn->weights[i] = malloc(weight_count * sizeof(float));
        nn->biases[i] = malloc(layer_sizes[i + 1] * sizeof(float));
        if (!nn->weights[i] || !nn->biases[i]) return NULL;

        // He initialization
        float scale = sqrtf(2.0f / layer_sizes[i]);
        for (size_t j = 0; j < weight_count; ++j) {
            float r = (float)rand() / RAND_MAX;
            nn->weights[i][j] = (r * 2.0f - 1.0f) * scale;
        }
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            float r = (float)rand() / RAND_MAX;
            nn->biases[i][j] = r;
        }
    }

    return nn;
}

void free_network(NeuralNetwork* nn) {
    if (!nn) return;

    for (int i = 0; i < nn->num_layers; ++i) {
        if (nn->outputs[i]) free(nn->outputs[i]);
        if (nn->errors[i]) free(nn->errors[i]);
    }

    for (int i = 0; i < nn->num_layers - 1; ++i) {
        if (nn->weights[i]) free(nn->weights[i]);
        if (nn->biases[i]) free(nn->biases[i]);
    }

    free(nn->layer_sizes);
    free(nn->weights);
    free(nn->biases);
    free(nn->outputs);
    free(nn->errors);
    free(nn);
}

void forward_pass(NeuralNetwork* nn, float* input) {
    // Copy input to first layer output
    memcpy(nn->outputs[0], input, nn->layer_sizes[0] * sizeof(float));

    for (int layer = 0; layer < nn->num_layers - 1; ++layer) {
        int in_size = nn->layer_sizes[layer];
        int out_size = nn->layer_sizes[layer + 1];
        float* weights = nn->weights[layer];
        float* biases = nn->biases[layer];
        float* in_out = nn->outputs[layer];
        float* out_out = nn->outputs[layer + 1];

        for (int j = 0; j < out_size; ++j) {
            float sum = biases[j];
            int weight_offset = j * in_size;
            for (int k = 0; k < in_size; ++k) {
                sum += weights[weight_offset + k] * in_out[k];
            }
            // ReLU for hidden layers, linear for output (will apply softmax later)
            if (layer < nn->num_layers - 2) {
                out_out[j] = (sum > 0) ? sum : 0;
            } else {
                out_out[j] = sum;
            }
        }
    }

    // Softmax on output layer
    int out_layer = nn->num_layers - 1;
    int out_size = nn->layer_sizes[out_layer];
    float* out = nn->outputs[out_layer];

    float max_val = out[0];
    for (int i = 1; i < out_size; ++i) {
        if (out[i] > max_val) max_val = out[i];
    }

    float sum = 0;
    for (int i = 0; i < out_size; ++i) {
        out[i] = expf(out[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < out_size; ++i) {
        out[i] /= sum;
    }
}

void backward_pass(NeuralNetwork* nn, uint8_t label) {
    int out_layer = nn->num_layers - 1;
    int out_size = nn->layer_sizes[out_layer];

    // Output layer error (cross-entropy with softmax: error = output - target)
    for (int i = 0; i < out_size; ++i) {
        nn->errors[out_layer][i] = nn->outputs[out_layer][i] - (i == label ? 1.0f : 0.0f);
    }

    // Backpropagate errors
    for (int layer = out_layer - 1; layer > 0; --layer) {
        int curr_size = nn->layer_sizes[layer];
        int next_size = nn->layer_sizes[layer + 1];
        float* weights = nn->weights[layer];
        float* curr_out = nn->outputs[layer];
        float* curr_err = nn->errors[layer];
        float* next_err = nn->errors[layer + 1];

        memset(curr_err, 0, curr_size * sizeof(float));
        for (int j = 0; j < next_size; ++j) {
            int weight_offset = j * curr_size;
            for (int k = 0; k < curr_size; ++k) {
                curr_err[k] += next_err[j] * weights[weight_offset + k];
            }
        }
        // ReLU derivative
        for (int k = 0; k < curr_size; ++k) {
            if (curr_out[k] <= 0) curr_err[k] = 0;
        }
    }

    // Update weights and biases
    for (int layer = 0; layer < nn->num_layers - 1; ++layer) {
        int in_size = nn->layer_sizes[layer];
        int out_size = nn->layer_sizes[layer + 1];
        float* weights = nn->weights[layer];
        float* biases = nn->biases[layer];
        float* in_out = nn->outputs[layer];
        float* out_err = nn->errors[layer + 1];

        for (int j = 0; j < out_size; ++j) {
            float lr_error = LEARNING_RATE * out_err[j];
            biases[j] -= lr_error;
            int weight_offset = j * in_size;
            for (int k = 0; k < in_size; ++k) {
                weights[weight_offset + k] -= lr_error * in_out[k];
            }
        }
    }
}

float test_accuracy(NeuralNetwork* nn, float* test_pixels, uint8_t* test_labels, int count) {
    int correct = 0;
    int px_size = IMAGE_ROWS * IMAGE_COLS;

    for (int img = 0; img < count; ++img) {
        forward_pass(nn, &test_pixels[img * px_size]);

        int out_layer = nn->num_layers - 1;
        int out_size = nn->layer_sizes[out_layer];
        float* out = nn->outputs[out_layer];

        int predicted = 0;
        float max_prob = out[0];
        for (int i = 1; i < out_size; ++i) {
            if (out[i] > max_prob) {
                max_prob = out[i];
                predicted = i;
            }
        }

        if (predicted == test_labels[img]) ++correct;
    }

    return (float)correct / count;
}

// ======================== TRAINING RUN ========================

typedef struct {
    int epochs_needed;
    float time_taken;
    float final_accuracy;
    bool achieved_100;
} RunResult;

RunResult train_until_accuracy(int* layer_sizes, int num_layers,
                               float* train_pixels, uint8_t* train_labels,
                               float* test_pixels, uint8_t* test_labels) {
    RunResult result = {EPOCHS_CAP, 0, 0.0f, false};
    int px_size = IMAGE_ROWS * IMAGE_COLS;

    NeuralNetwork* nn = create_network(layer_sizes, num_layers);
    if (!nn) {
        fprintf(stderr, "Failed to create network\n");
        return result;
    }

    clock_t start_time = clock();
    float accuracy = 0.0f;

    for (int epoch = 1; epoch <= EPOCHS_CAP; ++epoch) {
        // Train one epoch
        for (int img = 0; img < TRAIN_IMAGE_COUNT; ++img) {
            forward_pass(nn, &train_pixels[img * px_size]);
            backward_pass(nn, train_labels[img]);
        }

        // Test accuracy
        accuracy = test_accuracy(nn, test_pixels, test_labels, TEST_IMAGE_COUNT);

        if (accuracy >= TARGET_ACCURACY) {
            result.epochs_needed = epoch;
            result.achieved_100 = true;
            break;
        }
    }
    result.final_accuracy = accuracy * 100.0f;

    clock_t end_time = clock();
    result.time_taken = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    free_network(nn);
    return result;
}

// ======================== MAIN ARCHITECTURE SEARCH ========================

void log_to_csv_run(int* layer_sizes, int num_layers, float epochs_needed, float time_taken, float final_accuracy, bool achieved_100) {
    FILE* f = fopen("runs.csv", "a");
    if (!f) return;

    // Check if empty to write header
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "Architecture,EpochsNeeded,TimeTaken,FinalAccuracy,AchievedTargetAccuracy %.2f%%\n", TARGET_ACCURACY * 100.0f);
    }
    
    // Architecture string
    fprintf(f, "\"");
    for (int i = 0; i < num_layers; ++i) {
        fprintf(f, "%d", layer_sizes[i]);
        if (i < num_layers - 1) fprintf(f, ":");
    }
    fprintf(f, "\",%.2f,%.2fsec,%.2f%%,%s\n", 
            epochs_needed, 
            time_taken, 
            final_accuracy,
            achieved_100 ? "YES" : "NO");
    
    fclose(f);
}

void log_to_csv_result(int* layer_sizes, int num_layers, float avg_epochs, float avg_time, int achieved_count, int total_runs) {
    FILE* f = fopen("results.csv", "a");
    if (!f) return;

    // Check if empty to write header
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "Architecture,AvgEpochs,AvgTime,SuccessRate,Runs\n");
    }
    
    // Architecture string
    fprintf(f, "\"");
    for (int i = 0; i < num_layers; ++i) {
        fprintf(f, "%d", layer_sizes[i]);
        if (i < num_layers - 1) fprintf(f, ":");
    }
    fprintf(f, "\",%.2f,%.2f,%.2f%%,%d/%d\n", 
            avg_epochs, 
            avg_time, 
            ((float)achieved_count / total_runs) * 100.0f,
            achieved_count, total_runs);
    
    fclose(f);
}

void print_architecture(int* layer_sizes, int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        printf("%d", layer_sizes[i]);
        if (i < num_layers - 1) printf(" : ");
    }
}

int main() {
    srand((unsigned int)time(NULL));

    printf("Loading MNIST data...\n");

    float* train_pixels = NULL;
    uint8_t* train_labels = NULL;
    float* test_pixels = NULL;
    uint8_t* test_labels = NULL;

    if (!read_MNIST_images(TRAIN_IMAGES_PATH, &train_pixels)) {
        fprintf(stderr, "Error: Could not read train images\n");
        return 1;
    }
    if (!read_MNIST_labels(TRAIN_LABELS_PATH, &train_labels)) {
        fprintf(stderr, "Error: Could not read train labels\n");
        free(train_pixels);
        return 1;
    }
    if (!read_MNIST_images(TEST_IMAGES_PATH, &test_pixels)) {
        fprintf(stderr, "Error: Could not read test images\n");
        free(train_pixels); free(train_labels);
        return 1;
    }
    if (!read_MNIST_labels(TEST_LABELS_PATH, &test_labels)) {
        fprintf(stderr, "Error: Could not read test labels\n");
        free(train_pixels); free(train_labels); free(test_pixels);
        return 1;
    }

    printf("MNIST data loaded successfully.\n\n");
    printf("Starting architecture search...\n");
    printf("Hidden Layers: 0 to 10\n");
    printf("Nodes per layer: 1 to %d\n", NODES_CAP);
    printf("Runs per config: %d\n", RUNS_PER_CONFIG);
    printf("Max epochs per run: %d\n", EPOCHS_CAP);
    printf("Target accuracy: %.0f%%\n\n", TARGET_ACCURACY * 100);

    int layer_sizes[MAX_LAYERS];
    layer_sizes[0] = INPUT_NODES;

    // Iterate over number of hidden layers (0 to 10)
    for (int hidden_layers = 0; hidden_layers <= 10; ++hidden_layers) {
        int num_layers = 2 + hidden_layers; // Input + hidden + Output
        layer_sizes[num_layers - 1] = OUTPUT_NODES;

        // Initialize all hidden layers to 1 node
        for (int h = 0; h < hidden_layers; ++h) {
            layer_sizes[1 + h] = 1;
        }

        // If no hidden layers, just run the simple network
        if (hidden_layers == 0) {
            printf("\n========================================\n");
            printf("Testing: %d hidden layers\n", hidden_layers);
            printf("Architecture: ");
            print_architecture(layer_sizes, num_layers);
            printf("\n========================================\n");

            float total_epochs = 0;
            float total_time = 0;
            int achieved_count = 0;
            int runs_completed = 0;

            for (int run = 0; run < RUNS_PER_CONFIG; ++run) {
                RunResult res = train_until_accuracy(layer_sizes, num_layers,
                                                     train_pixels, train_labels,
                                                     test_pixels, test_labels);
                total_epochs += res.epochs_needed;
                total_time += res.time_taken;
                runs_completed++;
                if (res.achieved_100) ++achieved_count;
                printf("  Run %d: Epochs=%d, Time=%.2fs, Accuracy=%.2f%%, Achieved100=%s\n",
                       run + 1, res.epochs_needed, res.time_taken, res.final_accuracy,
                       res.achieved_100 ? "YES" : "NO");

                log_to_csv_run(layer_sizes, num_layers, res.epochs_needed, res.time_taken, res.final_accuracy, res.achieved_100);
                
                // Skip remaining runs if second run didn't achieve target accuracy
                if (run == 1 && achieved_count == 0) {
                    printf("  Second run failed target accuracy %.2f%% - skipping remaining runs\n", TARGET_ACCURACY * 100);
                    break;
                }
            }

            printf("\n------ RESULTS ------\n");
            printf("TOTAL EPOCHS AVERAGE : %.2f\n", total_epochs / runs_completed);
            printf("ARCHITECTURE : ");
            print_architecture(layer_sizes, num_layers);
            printf("\n");
            printf("TOTAL TIME AVERAGE : %.2f sec\n", total_time / runs_completed);
            printf("%.2f%% ACHIEVED : %d/%d runs\n", TARGET_ACCURACY * 100, achieved_count, runs_completed);
            
            log_to_csv_result(layer_sizes, num_layers, total_epochs / runs_completed, total_time / runs_completed, achieved_count, runs_completed);

            continue;
        }

        // Iterate by incrementing nodes layer by layer
        // Pattern: increment layer with fewest nodes first, then next, etc.
        // Stop when all layers reach NODES_CAP
        bool all_maxed = false;
        while (!all_maxed) {
            printf("\n========================================\n");
            printf("Testing: %d hidden layers\n", hidden_layers);
            printf("Architecture: ");
            print_architecture(layer_sizes, num_layers);
            printf("\n========================================\n");

            float total_epochs = 0;
            float total_time = 0;
            int achieved_count = 0;

            int runs_completed = 0;
            for (int run = 0; run < RUNS_PER_CONFIG; ++run) {
                RunResult res = train_until_accuracy(layer_sizes, num_layers,
                                                     train_pixels, train_labels,
                                                     test_pixels, test_labels);
                total_epochs += res.epochs_needed;
                total_time += res.time_taken;
                runs_completed++;
                if (res.achieved_100) ++achieved_count;
                printf("  Run %d: Epochs=%d, Time=%.2fs, Accuracy=%.2f%%, Achieved100=%s\n",
                       run + 1, res.epochs_needed, res.time_taken, res.final_accuracy,
                       res.achieved_100 ? "YES" : "NO");

                log_to_csv_run(layer_sizes, num_layers, res.epochs_needed, res.time_taken, res.final_accuracy, res.achieved_100);
                
                // Skip remaining runs if second run didn't achieve target accuracy
                if (run == 1 && achieved_count == 0) {
                    printf("  Second run failed target accuracy %.2f%% - skipping remaining runs\n", TARGET_ACCURACY * 100);
                    break;
                }
            }

            printf("\n------ RESULTS ------\n");
            printf("TOTAL EPOCHS AVERAGE : %.2f\n", total_epochs / runs_completed);
            printf("ARCHITECTURE : ");
            print_architecture(layer_sizes, num_layers);
            printf("\n");
            printf("TOTAL TIME AVERAGE : %.2f sec\n", total_time / runs_completed);
            printf("%.2f%% ACHIEVED : %d/%d runs\n", TARGET_ACCURACY * 100, achieved_count, runs_completed);

            log_to_csv_result(layer_sizes, num_layers, total_epochs / runs_completed, total_time / runs_completed, achieved_count, runs_completed);

            // Increment nodes: untill achieved target accuracy or all layers are maxed
            int carry = 1;
            for(int h = num_layers - 2; h > 0; --h) {
                layer_sizes[h] += carry;
                if(layer_sizes[h] >= NODES_CAP || achieved_count == RUNS_PER_CONFIG){
                    layer_sizes[h] = 1;
                    carry = 1;
                }else{
                    carry = 0;
                    break;
                }
            }
            if(carry == 1){
                all_maxed = true;
            }
        }
    }

    printf("\n\n========== ARCHITECTURE SEARCH COMPLETE ==========\n");

    free(train_pixels);
    free(train_labels);
    free(test_pixels);
    free(test_labels);

    return 0;
}