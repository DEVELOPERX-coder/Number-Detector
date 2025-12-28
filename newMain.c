#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TOTAL_NN_LAYERS 4
#define INPUT_NODES 784
#define OUPUT_NODES 10
#define H1_NODES 128
#define H2_NODES 64

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_COUNT 60000
#define TRAIN_IMAGE_COUNT 10000

uint32_t swap_endian(uint32_t val){
    return ((val << 24) & 0xFF000000) |
           ((val << 8) & 0x00FF0000) |
           ((val >> 8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

int read_mnist_training_images(const char *filename, uint8_t **pixels){
    FILE *file = fopen(filename,"rb");
    if(!file){
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 0;
    }

    uint32_t magic_number, num_images, num_rows, num_cols;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051) {
        fprintf(stderr, "Error: Invalid Magic Number %u (expected 2051)\n", magic_number);
        fclose(file);
        return 0;
    }

    fread(&num_images, sizeof(num_images), 1, file);
    num_images = swap_endian(num_images);

    fread(&num_rows, sizeof(num_rows), 1, file);
    num_rows = swap_endian(num_rows);

    fread(&num_cols, sizeof(num_cols), 1, file);
    num_cols = swap_endian(num_cols);

    if(num_rows != IMAGE_ROWS){
        fprintf(stderr, "IMAGES ROW MISMATCH");
        return 0;
    }

    if(num_cols != IMAGE_COLS){
        fprintf(stderr, "IMAGES COL MISMATCH");
        return 0;
    }

    size_t total_size = (size_t)num_images * num_rows * num_cols;
    *pixels = (uint8_t *)malloc(total_size);
    if (!pixels) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return 0;
    }

    fread(*pixels, 1, total_size, file);

    fclose(file);
    return 1;
}

int read_mnist_training_images_labels(const char* filename, uint8_t **train_images_labels){
    FILE *file = fopen(filename, "rb");
    if(!file){
        fprintf(stderr, "Error : Could not open file %s\n", filename);
        return 0;
    }

    uint32_t magic_number, num_labels;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = swap_endian(magic_number);

    if(magic_number != 2049){
        fprintf(stderr, "Error : Invalid Magic Number %u (expected 2049)", magic_number);
        fclose(file);
        return 0;
    }

    fread(&num_labels, sizeof(num_labels), 1, file);
    num_labels = swap_endian(num_labels);

    *train_images_labels = (uint8_t *)malloc(num_labels);
    if(!train_images_labels){
        fprintf(stderr, "Error : Memory allocation failed\n");
        fclose(file);
        return 0;
    }

    fread(*train_images_labels, 1, num_labels, file);

    fclose(file);
    return 1;
}

int main() {
    srand(time(NULL));

    uint8_t *pixels = NULL;

    if (read_mnist_training_images("./MNIST/train-images.idx3-ubyte", &pixels)) {
        printf("Successfully read %d images of size %dx%d\n", IMAGE_COUNT, IMAGE_ROWS, IMAGE_COLS);
    }else{
        return 1;
    }

    uint8_t *labels = NULL;

    if(read_mnist_training_images_labels("./MNIST/train-labels.idx1-ubyte", &labels)){
        printf("Successfully read %d labels\n", IMAGE_COUNT);
    }else{
        free(pixels);
        return 1;
    }

    uint8_t *train_pixels = NULL;

    if (read_mnist_training_images("./MNIST/t10k-images.idx3-ubyte", &train_pixels)) {
        printf("Successfully read %d train_images of size %dx%d\n", TRAIN_IMAGE_COUNT, IMAGE_ROWS, IMAGE_COLS);
    }else{
        free(pixels);
        free(labels);
        return 1;
    }

    uint8_t *train_labels = NULL;

    if(read_mnist_training_images_labels("./MNIST/t10k-labels.idx1-ubyte", &train_labels)){
        printf("Successfully read %d train_labels\n", TRAIN_IMAGE_COUNT);
    }else{
        free(pixels);
        free(labels);
        free(train_pixels);
        return 1;
    }

    float* weights_I_H1 = (float *)calloc(INPUT_NODES * H1_NODES, sizeof(float));
    if(!weights_I_H1){
        fprintf(stderr, "Error: Memory Allocation Error");
        free(pixels);
        free(labels);

        free(train_pixels);
        free(train_labels);

        return 1;
    }
    float* weights_H1_H2 = (float *)calloc(H1_NODES * H2_NODES, sizeof(float));
    if(!weights_H1_H2){
        fprintf(stderr, "Error: Memory Allocation Error");
        free(pixels);
        free(labels);

        free(train_pixels);
        free(train_labels);

        free(weights_I_H1);

        return 1;
    }
    float* weights_H2_O = (float *)calloc(H2_NODES * OUPUT_NODES, sizeof(float));
    if(!weights_H2_O){
        fprintf(stderr, "Error: Memory Allocation Error");
        free(pixels);
        free(labels);

        free(train_pixels);
        free(train_labels);

        free(weights_I_H1);
        free(weights_H1_H2);

        return 1;
    }

    for(int i = 0; i < INPUT_NODES * H1_NODES; ++i){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / INPUT_NODES);

        weights_I_H1[i] = (random_0_1 * 2.0f) - 1.0f;
        weights_I_H1[i] *= scale;
    }
    
    for(int i = 0; i < H1_NODES * H2_NODES; ++i){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);

        weights_H1_H2[i] = (random_0_1 * 2.0f) - 1.0f;
        weights_H1_H2[i] *= scale;
    }

    for(int i = 0; i < H2_NODES * OUPUT_NODES; ++i){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);

        weights_H2_O[i] = (random_0_1 * 2.0f) - 1.0f;
        weights_H2_O[i] *= scale;
    }

    float* bias_H1 = (float *)calloc(H1_NODES, sizeof(float));
    if(!bias_H1){
        fprintf(stderr, "Error: Memory Allocation Error");
        free(pixels);
        free(labels);

        free(train_pixels);
        free(train_labels);

        free(weights_I_H1);
        free(weights_H1_H2);
        free(weights_H2_O);

        return 1;
    }

    float* bias_H2 = (float *)calloc(H2_NODES, sizeof(float));
    if(!bias_H2){
        fprintf(stderr, "Error: Memory Allocation Error");
        free(pixels);
        free(labels);

        free(train_pixels);
        free(train_labels);

        free(weights_I_H1);
        free(weights_H1_H2);
        free(weights_H2_O);

        free(bias_H1);

        return 1;
    }

    float* bias_O = (float *)calloc(OUPUT_NODES, sizeof(float));
    if(!bias_O){
        fprintf(stderr, "Error: Memory Allocation Error");
        free(pixels);
        free(labels);

        free(train_pixels);
        free(train_labels);

        free(weights_I_H1);
        free(weights_H1_H2);
        free(weights_H2_O);

        free(bias_H1);
        free(bias_H2);

        return 1;
    }

    for(int i = 0; i < H1_NODES; ++i){
        float random_0_1 = (float)rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);

        bias_H1[i] = (random_0_1 * 2.0f) - 1.0f;
        bias_H1[i] *= scale;
    }

    for(int i = 0; i < H2_NODES; ++i){
        float random_0_1 = (float)rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);

        bias_H2[i] = (random_0_1 * 2.0f) - 1.0f;
        bias_H2[i] *= scale;
    }

    for(int i = 0; i < OUPUT_NODES; ++i){
        float random_0_1 = (float)rand() / RAND_MAX;
        float scale = sqrt(2.0f / OUPUT_NODES);

        bias_O[i] = (random_0_1 * 2.0f) - 1.0f;
        bias_O[i] *= scale;
    }

    float correct_probability = 0;
    int correct = 0;
    int total = 0;

    printf("TRAINING STARTED!\n");

    clock_t start_time = clock();

    for(int i = 0; i < IMAGE_COUNT; ++i){
        //++total;
        int index = i * IMAGE_ROWS * IMAGE_COLS;

        float* H1_OUTPUT = (float *)calloc(H1_NODES, sizeof(float));
        for(int j = 0; j < H1_NODES; ++j){
            int ind = 0;
            for(int k = j * INPUT_NODES; k < j * INPUT_NODES + INPUT_NODES; ++k){
                H1_OUTPUT[j] += weights_I_H1[k] * (float)(pixels[index + ind++]) / 255;
            }
            H1_OUTPUT[j] += bias_H1[j];

            if(H1_OUTPUT[j] <= 0) H1_OUTPUT[j] = 0;
        }

        float* H2_OUTPUT = (float *)calloc(H2_NODES, sizeof(float));
        for(int j = 0; j < H2_NODES; ++j){
            int ind = 0;
            for(int k = j * H1_NODES; k < j * H1_NODES + H1_NODES; ++k)
            H2_OUTPUT[j] += weights_H1_H2[k] * H1_OUTPUT[ind++];

            H2_OUTPUT[j] += bias_H2[j];

            if(H2_OUTPUT[j] <= 0) H2_OUTPUT[j] = 0;
        }

        float* O_OUTPUT = (float *)calloc(OUPUT_NODES, sizeof(float));
        for(int j = 0; j < OUPUT_NODES; ++j){
            int ind = 0;
            for(int k = j * H2_NODES; k < j * H2_NODES + H2_NODES; ++k){
                O_OUTPUT[j] += weights_H2_O[k] * H2_OUTPUT[ind++];
            }
            O_OUTPUT[j] += bias_O[j];
        }

        float sum = 0;

        for(int j = 0; j < OUPUT_NODES; ++j){
            sum += exp(O_OUTPUT[j]);
        }
        for(int j = 0; j < OUPUT_NODES; ++j){
            O_OUTPUT[j] = exp(O_OUTPUT[j]) / sum;
        }

        float learning_rate = 0.01;
        uint8_t correct_label = labels[i];
        float* expected_labels = (float*)calloc(OUPUT_NODES, sizeof(float));
        expected_labels[correct_label] = 1;

        //if(O_OUTPUT[correct_label] > 0.5f) ++correct;

        float* error = (float*)calloc(OUPUT_NODES, sizeof(float));
        for(int j = 0; j < OUPUT_NODES; ++j){
            error[j] = O_OUTPUT[j] - expected_labels[j];
        }

        float* error_H2 = (float*)calloc(H2_NODES, sizeof(float));
        for(int j = 0; j < OUPUT_NODES; ++j){
            int ind = 0;
            for(int k = j * H2_NODES; k < j * H2_NODES + H2_NODES; ++k){
                error_H2[ind] += error[j] * weights_H2_O[k];
                ++ind;
            }
        }
        for(int j = 0; j < H2_NODES; ++j){
            if(H2_OUTPUT[j] == 0) error_H2[j] = 0;
        }

        for(int j = 0; j < OUPUT_NODES; ++j){
            bias_O[j] = bias_O[j] - learning_rate * error[j];
        }

        for(int j = 0; j < OUPUT_NODES; ++j){
            int ind = 0;
            for(int k = j * H2_NODES; k < j * H2_NODES + H2_NODES; ++k){
                weights_H2_O[k] = weights_H2_O[k] - learning_rate * error[j] * H2_OUTPUT[ind++];
            }
        }

        float* error_H1 = (float*)calloc(H1_NODES, sizeof(float));
        for(int j = 0; j < H2_NODES; ++j){
            int ind = 0;
            for(int k = j * H1_NODES; k < j * H1_NODES + H1_NODES; ++k){
                error_H1[ind++] += error_H2[j] * weights_H1_H2[k];
            }
        }

        for(int j = 0; j < H1_NODES; ++j){
            if(H1_OUTPUT[j] == 0){
                error_H1[j] = 0;
            }
        }

        for(int j = 0; j < H2_NODES; ++j){
            bias_H2[j] = bias_H2[j] - learning_rate * error_H2[j];
        }

        for(int j = 0; j < H2_NODES; ++j){
            int ind = 0;
            for(int k = j * H1_NODES; k < j * H1_NODES + H1_NODES; ++k){
                weights_H1_H2[k] = weights_H1_H2[k] - learning_rate * error_H2[j] * H1_OUTPUT[ind++];
            }
        }

        for(int j = 0; j < H1_NODES; ++j){
            bias_H1[j] = bias_H1[j] - learning_rate * error_H1[j];
        }

        for(int j = 0; j < H1_NODES; ++j){
            int ind = 0;
            for(int k = j * INPUT_NODES; k < j * INPUT_NODES + INPUT_NODES; ++k){
                weights_I_H1[k] = weights_I_H1[k] - learning_rate * error_H1[j] * (float)pixels[i * IMAGE_COLS * IMAGE_ROWS + ind] / 255;
                ++ind;
            }
        }

        free(H1_OUTPUT);
        free(H2_OUTPUT);
        free(O_OUTPUT);
        free(expected_labels);
        free(error);
        free(error_H2);
        free(error_H1);

        // if(i % 1000 == 0) printf("%f : probability check for %d : images \n", (float) correct / total, total);
    }

    clock_t end_time = clock();

    printf("Training completed! Time : %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    correct_probability = 0;
    correct = 0;
    total = 0;

    printf("Now Checking Model Accuracy!\n");

    for(int i = 0; i < TRAIN_IMAGE_COUNT; ++i){
        ++total;
        int index = i * IMAGE_ROWS * IMAGE_COLS;

        float* H1_OUTPUT = (float *)calloc(H1_NODES, sizeof(float));
        for(int j = 0; j < H1_NODES; ++j){
            int ind = 0;
            for(int k = j * INPUT_NODES; k < j * INPUT_NODES + INPUT_NODES; ++k){
                H1_OUTPUT[j] += weights_I_H1[k] * (float)(train_pixels[index + ind++]) / 255;
            }
            H1_OUTPUT[j] += bias_H1[j];

            if(H1_OUTPUT[j] <= 0) H1_OUTPUT[j] = 0;
        }

        float* H2_OUTPUT = (float *)calloc(H2_NODES, sizeof(float));
        for(int j = 0; j < H2_NODES; ++j){
            int ind = 0;
            for(int k = j * H1_NODES; k < j * H1_NODES + H1_NODES; ++k)
            H2_OUTPUT[j] += weights_H1_H2[k] * H1_OUTPUT[ind++];

            H2_OUTPUT[j] += bias_H2[j];

            if(H2_OUTPUT[j] <= 0) H2_OUTPUT[j] = 0;
        }

        float* O_OUTPUT = (float *)calloc(OUPUT_NODES, sizeof(float));
        for(int j = 0; j < OUPUT_NODES; ++j){
            int ind = 0;
            for(int k = j * H2_NODES; k < j * H2_NODES + H2_NODES; ++k){
                O_OUTPUT[j] += weights_H2_O[k] * H2_OUTPUT[ind++];
            }
            O_OUTPUT[j] += bias_O[j];
        }

        float sum = 0;

        for(int j = 0; j < OUPUT_NODES; ++j){
            sum += exp(O_OUTPUT[j]);
        }
        for(int j = 0; j < OUPUT_NODES; ++j){
            O_OUTPUT[j] = exp(O_OUTPUT[j]) / sum;
        }

        float learning_rate = 0.01;
        uint8_t correct_label = train_labels[i];
        float* expected_labels = (float*)calloc(OUPUT_NODES, sizeof(float));
        expected_labels[correct_label] = 1;

        //if(O_OUTPUT[correct_label] > 0.5f) ++correct;
        int greatest = 1;
        for(int j = 0; j < OUPUT_NODES; ++j){
            if(j != correct_label && O_OUTPUT[correct_label] <= O_OUTPUT[j]){
                greatest = 0;
                break;
            }
        }
        if(greatest == 1) ++correct;

        free(H1_OUTPUT);
        free(H2_OUTPUT);
        free(O_OUTPUT);
        free(expected_labels);

        // if(i % 1000 == 0) printf("%f : probability check for %d : images \n", (float) correct / total, total);
    }

    printf("%f : Probability Correct", (float)correct / total);

    free(pixels);
    free(labels);

    free(train_pixels);
    free(train_labels);

    free(weights_I_H1);
    free(weights_H1_H2);
    free(weights_H2_O);

    free(bias_H1);
    free(bias_H2);
    free(bias_O);

    return 0;
}