#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define TOTAL_NN_LAYERS 4
#define INPUT_NODES 784
#define OUPUT_NODES 10
#define H1_NODES 128
#define H2_NODES 64

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_COUNT 60000
#define TRAIN_IMAGE_COUNT 10000

#define EPOCHS 10
#define BATCH_SIZE 30

uint8_t* train_pixels = NULL;
uint8_t* train_labels = NULL;
uint8_t* test_pixels = NULL;
uint8_t* test_labels = NULL;

float* weights_I_H1 = NULL;
float* weights_H1_H2 = NULL;
float* weights_H2_O = NULL;

float* bias_H1 = NULL;
float* bias_H2 = NULL;
float* bias_O = NULL;

float* H1_OUTPUT = NULL;
float* H2_OUTPUT = NULL;
float* O_OUTPUT = NULL;

float* error = NULL;
float* error_H2 = NULL;
float* error_H1 = NULL;

void terminate_program(int check_condition){
    if(train_pixels != NULL) free(train_pixels);
    if(train_labels != NULL) free(train_labels);

    if(test_pixels != NULL) free(test_pixels);
    if(test_labels != NULL) free(test_labels);

    if(weights_I_H1 != NULL) free(weights_I_H1);
    if(weights_H1_H2 != NULL) free(weights_H1_H2);
    if(weights_H2_O != NULL) free(weights_H2_O);

    if(bias_H1 != NULL) free(bias_H1);
    if(bias_H2 != NULL) free(bias_H2);
    if(bias_O != NULL) free(bias_O);

    if(H1_OUTPUT != NULL) free(H1_OUTPUT);
    if(H2_OUTPUT != NULL) free(H2_OUTPUT);
    if(O_OUTPUT != NULL) free(O_OUTPUT);

    if(error != NULL) free(error);
    if(error_H2 != NULL) free(error_H2);
    if(error_H1 != NULL) free(error_H1);
    
    if(check_condition == 1) exit(EXIT_FAILURE);
    else exit(EXIT_SUCCESS);
}

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
        terminate_program(1);
    }

    uint32_t magic_number, num_images, num_rows, num_cols;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051) {
        fprintf(stderr, "Error: Invalid Magic Number %u (expected 2051)\n", magic_number);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fread(&num_images, sizeof(num_images), 1, file);
    num_images = swap_endian(num_images);

    fread(&num_rows, sizeof(num_rows), 1, file);
    num_rows = swap_endian(num_rows);

    fread(&num_cols, sizeof(num_cols), 1, file);
    num_cols = swap_endian(num_cols);

    if(num_rows != IMAGE_ROWS){
        fprintf(stderr, "IMAGES ROW MISMATCH");
        exit(EXIT_FAILURE);
    }

    if(num_cols != IMAGE_COLS){
        fprintf(stderr, "IMAGES COL MISMATCH");
        exit(EXIT_FAILURE);
    }

    size_t total_size = (size_t)num_images * num_rows * num_cols;
    *pixels = malloc(total_size);
    if (!pixels) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        terminate_program(1);
    }

    fread(*pixels, 1, total_size, file);

    fclose(file);
    return 1;
}

int read_mnist_training_images_labels(const char* filename, uint8_t **train_images_labels){
    FILE *file = fopen(filename, "rb");
    if(!file){
        fprintf(stderr, "Error : Could not open file %s\n", filename);
        terminate_program(1);
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

    *train_images_labels = malloc(num_labels);
    if(!train_images_labels){
        fprintf(stderr, "Error : Memory allocation failed\n");
        fclose(file);
        terminate_program(1);
    }

    fread(*train_images_labels, 1, num_labels, file);

    fclose(file);
    return 1;
}

int main() {
    srand(time(NULL));

    if (read_mnist_training_images("./MNIST/train-images.idx3-ubyte", &train_pixels)) {
        printf("Successfully read %d images of size %dx%d\n", IMAGE_COUNT, IMAGE_ROWS, IMAGE_COLS);
    }else{
        terminate_program(1);
    }

    if(read_mnist_training_images_labels("./MNIST/train-labels.idx1-ubyte", &train_labels)){
        printf("Successfully read %d labels\n", IMAGE_COUNT);
    }else{
        terminate_program(1);
    }

    if (read_mnist_training_images("./MNIST/t10k-images.idx3-ubyte", &test_pixels)) {
        printf("Successfully read %d train_images of size %dx%d\n", TRAIN_IMAGE_COUNT, IMAGE_ROWS, IMAGE_COLS);
    }else{
        terminate_program(1);
    }

    if(read_mnist_training_images_labels("./MNIST/t10k-labels.idx1-ubyte", &test_labels)){
        printf("Successfully read %d train_labels\n", TRAIN_IMAGE_COUNT);
    }else{
        terminate_program(1);
    }

    weights_I_H1 = calloc(INPUT_NODES * H1_NODES, sizeof(float));
    if(!weights_I_H1){
        fprintf(stderr, "Error: Memory Allocation Error");
        terminate_program(1);
    }
    weights_H1_H2 = calloc(H1_NODES * H2_NODES, sizeof(float));
    if(!weights_H1_H2){
        fprintf(stderr, "Error: Memory Allocation Error");
        terminate_program(1);
    }
    weights_H2_O = calloc(H2_NODES * OUPUT_NODES, sizeof(float));
    if(!weights_H2_O){
        fprintf(stderr, "Error: Memory Allocation Error");
        terminate_program(1);
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

    bias_H1 = calloc(H1_NODES, sizeof(float));
    if(!bias_H1){
        fprintf(stderr, "Error: Memory Allocation Error");
        terminate_program(1);
    }

    bias_H2 = calloc(H2_NODES, sizeof(float));
    if(!bias_H2){
        fprintf(stderr, "Error: Memory Allocation Error");
        terminate_program(1);
    }

    bias_O = calloc(OUPUT_NODES, sizeof(float));
    if(!bias_O){
        fprintf(stderr, "Error: Memory Allocation Error");
        terminate_program(1);
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

    H1_OUTPUT = calloc(H1_NODES, sizeof(float));
    if(!H1_OUTPUT){
        terminate_program(1);
    }
    H2_OUTPUT = calloc(H2_NODES, sizeof(float));
    if(!H2_OUTPUT){
        terminate_program(1);
    }
    O_OUTPUT = calloc(OUPUT_NODES, sizeof(float));
    if(!O_OUTPUT){
        terminate_program(1);
    }

    error = calloc(OUPUT_NODES, sizeof(float));
    if(!error){
        terminate_program(1);
    }
    error_H2 = calloc(H2_NODES, sizeof(float));
    if(!error_H2){
        terminate_program(1);
    }
    error_H1 = calloc(H1_NODES, sizeof(float));
    if(!error_H1){
        terminate_program(1);
    }

    float correct_probability = 0;
    int correct = 0;
    int total = 0;

    printf("TRAINING STARTED!\n");

    clock_t start_time = clock();

    for(int i = 0; i < IMAGE_COUNT; ++i){
        int index = i * IMAGE_ROWS * IMAGE_COLS;

        for(int j = 0; j < H1_NODES; ++j){
            H1_OUTPUT[j] = 0;
            int ind = 0;
            for(int k = j * INPUT_NODES; k < j * INPUT_NODES + INPUT_NODES; ++k){
                H1_OUTPUT[j] += weights_I_H1[k] * (float)(train_pixels[index + ind++]) / 255;
            }
            H1_OUTPUT[j] += bias_H1[j];

            if(H1_OUTPUT[j] <= 0) H1_OUTPUT[j] = 0;
        }

        for(int j = 0; j < H2_NODES; ++j){
            H2_OUTPUT[j] = 0;
            int ind = 0;
            for(int k = j * H1_NODES; k < j * H1_NODES + H1_NODES; ++k)
            H2_OUTPUT[j] += weights_H1_H2[k] * H1_OUTPUT[ind++];

            H2_OUTPUT[j] += bias_H2[j];

            if(H2_OUTPUT[j] <= 0) H2_OUTPUT[j] = 0;
        }

        for(int j = 0; j < OUPUT_NODES; ++j){
            O_OUTPUT[j] = 0;
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

        for(int j = 0; j < OUPUT_NODES; ++j){
            error[j] = 0;
            error[j] = O_OUTPUT[j] - expected_labels[j];
        }

        memset(error_H2, 0, H2_NODES * sizeof(float));
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

        memset(error_H1, 0, H1_NODES * sizeof(float));
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
                weights_I_H1[k] = weights_I_H1[k] - learning_rate * error_H1[j] * (float)train_pixels[i * IMAGE_COLS * IMAGE_ROWS + ind] / 255;
                ++ind;
            }
        }

        free(expected_labels);
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

        for(int j = 0; j < H1_NODES; ++j){
            H1_OUTPUT[j] = 0;
            int ind = 0;
            for(int k = j * INPUT_NODES; k < j * INPUT_NODES + INPUT_NODES; ++k){
                H1_OUTPUT[j] += weights_I_H1[k] * (float)(train_pixels[index + ind++]) / 255;
            }
            H1_OUTPUT[j] += bias_H1[j];

            if(H1_OUTPUT[j] <= 0) H1_OUTPUT[j] = 0;
        }

        for(int j = 0; j < H2_NODES; ++j){
            H2_OUTPUT[j] = 0;
            int ind = 0;
            for(int k = j * H1_NODES; k < j * H1_NODES + H1_NODES; ++k)
            H2_OUTPUT[j] += weights_H1_H2[k] * H1_OUTPUT[ind++];

            H2_OUTPUT[j] += bias_H2[j];

            if(H2_OUTPUT[j] <= 0) H2_OUTPUT[j] = 0;
        }

        for(int j = 0; j < OUPUT_NODES; ++j){
            O_OUTPUT[j] = 0;
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

        int greatest = 1;
        for(int j = 0; j < OUPUT_NODES; ++j){
            if(j != correct_label && O_OUTPUT[correct_label] <= O_OUTPUT[j]){
                greatest = 0;
                break;
            }
        }
        if(greatest == 1) ++correct;

        free(expected_labels);
    }

    printf("%f : Probability Correct", (float)correct / total);

    terminate_program(0);
}