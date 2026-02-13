#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>

#define TRAIN_IMAGES_PATH "../../MNIST/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "../../MNIST/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "../../MNIST/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "../../MNIST/t10k-labels.idx1-ubyte"

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define TRAIN_IMAGE_COUNT 60000
#define TEST_IMAGE_COUNT 10000

#define INPUT_NODES 784
#define OUTPUT_NODES 10
#define LEARNING_RATE 0.01f

uint32_t swap_endian(uint32_t value) {
    return ((value >> 24) & 0x000000FF) |
           ((value >> 8)  & 0x0000FF00) |
           ((value << 8)  & 0x00FF0000) |
           ((value << 24) & 0xFF000000);
}

void load_mnist_images(const char* filepath, double** pixels, int count) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        perror("Error opening image file");
        exit(EXIT_FAILURE);
    }

    uint32_t magic, num_images, rows, cols;

    fread(&magic, sizeof(magic), 1, file);
    fread(&num_images, sizeof(num_images), 1, file);
    fread(&rows, sizeof(rows), 1, file);
    fread(&cols, sizeof(cols), 1, file);

    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051 || num_images != count || rows != IMAGE_ROWS || cols != IMAGE_COLS) {
        fprintf(stderr, "Error: Invalid MNIST image file format\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    uint8_t* images_data = (uint8_t*)malloc(count * IMAGE_ROWS * IMAGE_COLS * sizeof(uint8_t));
    if (!images_data) {
        perror("Error allocating memory for images data");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fread(images_data, sizeof(uint8_t), count * IMAGE_ROWS * IMAGE_COLS, file);

    *pixels = (double*)malloc(count * IMAGE_ROWS * IMAGE_COLS * sizeof(double));
    if (!*pixels) {
        perror("Error allocating memory for pixels");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < IMAGE_ROWS * IMAGE_COLS; j++) {
            (*pixels)[i * IMAGE_ROWS * IMAGE_COLS + j] = (double)images_data[i * IMAGE_ROWS * IMAGE_COLS + j] / 255.0f;
        }
    }

    free(images_data);
    fclose(file);
}

void load_mnist_labels(const char* filepath, uint8_t** labels, int count) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        perror("Error opening label file");
        exit(EXIT_FAILURE);
    }

    uint32_t magic, num_labels;

    fread(&magic, sizeof(magic), 1, file);
    fread(&num_labels, sizeof(num_labels), 1, file);

    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    if (magic != 2049 || num_labels != count) {
        fprintf(stderr, "Error: Invalid MNIST label file format\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    *labels = (uint8_t*)malloc(count * sizeof(uint8_t));
    if (!*labels) {
        perror("Error allocating memory for labels");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fread(*labels, sizeof(uint8_t), count, file);

    fclose(file);
}

int main(){
    srand(time(NULL));

    double* train_pixels = NULL;
    uint8_t* train_labels = NULL;

    double* test_pixels = NULL;
    uint8_t* test_labels = NULL;

    load_mnist_images(TRAIN_IMAGES_PATH, &train_pixels, TRAIN_IMAGE_COUNT);
    load_mnist_labels(TRAIN_LABELS_PATH, &train_labels, TRAIN_IMAGE_COUNT);

    load_mnist_images(TEST_IMAGES_PATH, &test_pixels, TEST_IMAGE_COUNT);
    load_mnist_labels(TEST_LABELS_PATH, &test_labels, TEST_IMAGE_COUNT);

    printf("Loaded %d training images and %d test images\n", TRAIN_IMAGE_COUNT, TEST_IMAGE_COUNT);
    
    double* weights = (double*)malloc(INPUT_NODES * OUTPUT_NODES * sizeof(double));
    if (!weights) {
        perror("Error allocating memory for weights");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < INPUT_NODES * OUTPUT_NODES; i++) {
        double random = (double) rand() / RAND_MAX;
        double scale = sqrt(2.0 / (double)INPUT_NODES);
        weights[i] = (random * 2.0 - 1.0) * scale;
    }

    double* biases = (double*)malloc(OUTPUT_NODES * sizeof(double));
    if (!biases) {
        perror("Error allocating memory for biases");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        double random = (double) rand() / RAND_MAX;
        double scale = sqrt(2.0 / (double)OUTPUT_NODES);
        biases[i] = (random * 2.0 - 1.0) * scale;
    }

    double* output = (double*)malloc(OUTPUT_NODES * sizeof(double));
    if (!output) {
        perror("Error allocating memory for output");
        exit(EXIT_FAILURE);
    }

    double total_time = 0;

    for(int EPOCH = 1; EPOCH <= 1; ++EPOCH){
        int correct = 0;
        int total = 0;
        
        clock_t start_time = clock();

        for(int image_index = 0; image_index < TRAIN_IMAGE_COUNT; ++image_index){
            // Forward Propagation
            int predicted_label = 0;

            for(int i = 0; i < OUTPUT_NODES; ++i){
                output[i] = 0;
                for(int j = 0; j < INPUT_NODES; ++j){
                    output[i] += train_pixels[image_index * INPUT_NODES + j] * weights[i * INPUT_NODES + j];
                }
                output[i] += biases[i];

                if(output[i] < 0) output[i] = 0;

                if(output[i] > output[predicted_label]){
                    predicted_label = i;
                }
            }

            double min = LDBL_MAX;
            for(int node_index = 0; node_index < OUTPUT_NODES; ++node_index){
                if(output[node_index] < min){
                    min = output[node_index];
                }
            }
            double sum = 0;
            for(int node_index = 0; node_index < OUTPUT_NODES; ++node_index){
                output[node_index] -= min;
                output[node_index] = expf(output[node_index]);
                sum += output[node_index];
            }
            for(int node_index = 0; node_index < OUTPUT_NODES; ++node_index){
                output[node_index] /= sum;
            }

            // Backward Propagation
            double* error = (double*)malloc(OUTPUT_NODES * sizeof(double));
            if (!error) {
                perror("Error allocating memory for error");
                exit(EXIT_FAILURE);
            }

            for(int i = 0; i < OUTPUT_NODES; ++i){
                if(output[i] > 0) error[i] = output[i] - (train_labels[image_index] == i ? 1 : 0);
                else error[i] = 0;
            }

            for(int i = 0; i < OUTPUT_NODES; ++i){
                for(int j = 0; j < INPUT_NODES; ++j){
                    weights[i * INPUT_NODES + j] -= error[i] * train_pixels[image_index * INPUT_NODES + j] * LEARNING_RATE;
                }
                biases[i] -= error[i] * LEARNING_RATE;
            }

            if(predicted_label == train_labels[image_index]){
                ++correct;
            }
            ++total;
        }

        clock_t end_time = clock();
        double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        double accuracy = (double)correct / total;
        printf("Epoch %d: Accuracy = %f, Time = %f\n", EPOCH, accuracy, time_taken);
    }

    FILE* model_file = fopen("data/model.bin", "wb");
    if (model_file) {
        int input_nodes = INPUT_NODES;
        int output_nodes = OUTPUT_NODES;
        fwrite(&input_nodes, sizeof(int), 1, model_file);
        fwrite(&output_nodes, sizeof(int), 1, model_file);
        fwrite(weights, sizeof(double), INPUT_NODES * OUTPUT_NODES, model_file);
        fwrite(biases, sizeof(double), OUTPUT_NODES, model_file);
        fclose(model_file);
        printf("Model saved to data/model.bin\n");
    } else {
        perror("Error opening data/model.bin for writing");
    }

    int correct = 0;
    int total = 0;
    
    int class_correct[OUTPUT_NODES];
    int class_total[OUTPUT_NODES];
    for(int i = 0; i < OUTPUT_NODES; ++i) {
        class_correct[i] = 0;
        class_total[i] = 0;
    }

    for(int image_index = 0; image_index < TEST_IMAGE_COUNT; ++image_index){
        // Forward Propagation
        int predicted_label = 0;

        for(int i = 0; i < OUTPUT_NODES; ++i){
            output[i] = 0;
            for(int j = 0; j < INPUT_NODES; ++j){
                output[i] += test_pixels[image_index * INPUT_NODES + j] * weights[i * INPUT_NODES + j];
            }
            output[i] += biases[i];

            if(output[i] > output[predicted_label]){
                predicted_label = i;
            }
        }

        uint8_t actual_label = test_labels[image_index];
        if(predicted_label == actual_label){
            ++correct;
            class_correct[actual_label]++;
        }
        class_total[actual_label]++;
        ++total;
    }

    double accuracy = (double)correct / total;
    printf("Test Accuracy = %f\n", accuracy);
    
    double class_accuracies[OUTPUT_NODES];
    for(int i=0; i<OUTPUT_NODES; ++i){
        class_accuracies[i] = (class_total[i] > 0) ? (double)class_correct[i] / class_total[i] : 0.0;
        printf("Digit %d Accuracy: %.2f%%\n", i, class_accuracies[i]*100);
    }

    // Re-open/Append accuracies to model file
    // Note: We already closed it, or rather we should have kept it open or reopened it.
    // The previous block closed it. Let's reopen in append binary mode 'ab' or just write it earlier.
    // Actually, generating the model file happens BEFORE testing in the original code.
    // I need to move the file writing AFTER testing or reopen it.
    model_file = fopen("data/model.bin", "ab"); // Append binary
    if (model_file) {
        fwrite(class_accuracies, sizeof(double), OUTPUT_NODES, model_file);
        fclose(model_file);
        printf("Updated model.bin with class accuracies.\n");
    } else {
         perror("Error opening data/model.bin for appending");
    }


    free(train_pixels);
    free(train_labels);
    free(test_pixels);
    free(test_labels);
    free(weights);
    free(biases);
    free(output);
    return 0;
}