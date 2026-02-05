#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>

#define EPOCHS 10
#define TOTAL_NN_LAYERS 4
#define INPUT_NODES 784
#define H1_NODES 128
#define H2_NODES 64
#define OUTPUT_NODES 10
#define LEARNING_RATE 0.01f

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define TRAIN_IMAGE_COUNT 60000
#define TEST_IMAGE_COUNT 10000

void terminate_program(int check_condition, float* train_pixels, uint8_t* train_labels, float* test_pixels, uint8_t* test_labels, float* weights_I_H1, float* weights_H1_H2, float* weights_H2_O, float* bias_H1, float* bias_H2, float* bias_O, float* H1_OUTPUT, float* H2_OUTPUT, float* O_OUTPUT, float* error_O, float* error_H2, float* error_H1){
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

    if(error_O != NULL) free(error_O);
    if(error_H2 != NULL) free(error_H2);
    if(error_H1 != NULL) free(error_H1);
    
    if(check_condition == 1) exit(EXIT_FAILURE);
    else exit(EXIT_SUCCESS);
}

uint32_t swap_endian(uint32_t val){
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF) ;
}

int read_MNIST_images(const char* filename, float** pixels_float){
    FILE* file = fopen(filename, "rb");
    if(file == NULL){
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 1;
    }

    uint32_t magic_number, num_of_images, num_of_rows, num_of_cols;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = swap_endian(magic_number);
    if(magic_number != 2051){
        fprintf(stderr, "Error: Invalid Magic Number %u (expected 2051)\n", magic_number);
        fclose(file);
        return 1;
    }

    fread(&num_of_images, sizeof(num_of_images), 1, file);
    num_of_images = swap_endian(num_of_images);

    fread(&num_of_rows, sizeof(num_of_rows), 1, file);
    num_of_rows = swap_endian(num_of_rows);

    fread(&num_of_cols, sizeof(num_of_cols), 1, file);
    num_of_cols = swap_endian(num_of_cols);

    if(num_of_rows != IMAGE_ROWS){
        fprintf(stderr, "IMAGES ROW MISMATCH : %s filename\n", filename);
        return 1;
    }

    if(num_of_cols != IMAGE_COLS){
        fprintf(stderr, "IMAGES COL MISMATCH : %s filename\n", filename);
        return 1;
    }

    size_t total_size = (size_t)num_of_images * num_of_rows * num_of_cols;
    uint8_t* pixels = malloc(total_size);
    if(pixels == NULL){
        fprintf(stderr, "Error: Pixels Memory Allocation Failed : %s filename\n",filename);
        fclose(file);
        return 1;
    }

    fread(pixels, 1, total_size, file);
    fclose(file);

    *pixels_float = malloc(total_size * sizeof(float));
    if(*pixels_float == NULL){
        fprintf(stderr, "Error: Pixels_Float Memory Allocation Failed : %s filename\n",filename);
        return 1;
    }
    for(int i = 0; i < total_size; ++i){
        (*pixels_float)[i] = (float)pixels[i] / 255;
    }

    free(pixels);
    return 0;
}

int read_MNIST_labels(const char* filename, uint8_t** lables){
    FILE* file = fopen(filename, "rb");
    if(file == NULL){
        fprintf(stderr, "Error : Could not open file %s\n", filename);
        return 1;
    }

    uint32_t magic_number, num_of_labels;
    
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = swap_endian(magic_number);
    if(magic_number != 2049){
        fprintf(stderr, "Error : Invalid Magic Number %u (expected 2049)\n", magic_number);
        fclose(file);
        return 1;
    }

    fread(&num_of_labels, sizeof(num_of_labels), 1, file);
    num_of_labels = swap_endian(num_of_labels);

    *lables = malloc(num_of_labels);
    if(*lables == NULL){
        fprintf(stderr, "Error : Labels Memory Allocation Failed : %s filename\n", filename);
        fclose(file);
        return 1;
    }
    fread(*lables, 1, num_of_labels, file);

    fclose(file);
    return 0;
}

int main(){
    srand(time(NULL));

    float* train_pixels = NULL;
    uint8_t* train_labels = NULL;

    float* test_pixels = NULL;
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

    float* error_O = NULL;
    float* error_H2 = NULL;
    float* error_H1 = NULL;

    // Train

    if(read_MNIST_images("../MNIST/train-images.idx3-ubyte", &train_pixels)){
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }else{
        printf("Successfully Read %d images of size R %d X C %d\n", TRAIN_IMAGE_COUNT, IMAGE_ROWS, IMAGE_COLS);
    }

    if(read_MNIST_labels("../MNIST/train-labels.idx1-ubyte", &train_labels)){
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }else{
        printf("Successfully Read %d labels\n", TRAIN_IMAGE_COUNT);
    }

    // Testing

    if(read_MNIST_images("../MNIST/t10k-images.idx3-ubyte", &test_pixels)){
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }else{
        printf("Successfully Read %d images of size R %d X C %d\n", TEST_IMAGE_COUNT, IMAGE_ROWS, IMAGE_COLS);
    }

    if(read_MNIST_labels("../MNIST/t10k-labels.idx1-ubyte", &test_labels)){
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }else{
        printf("Successfully Read %d labels\n", TEST_IMAGE_COUNT);
    }

    // Memory Allocation

    weights_I_H1 = malloc((size_t)INPUT_NODES * H1_NODES * sizeof(float));
    if(weights_I_H1 == NULL){
        fprintf(stderr, "Error W_I_H1 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }
    for(int index = 0; index < INPUT_NODES * H1_NODES; ++index){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / INPUT_NODES);

        weights_I_H1[index] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    weights_H1_H2 = malloc((size_t)H1_NODES * H2_NODES * sizeof(float));
    if(weights_H1_H2 == NULL){
        fprintf(stderr, "Error W_H1_H2 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }
    for(int index = 0; index < H1_NODES * H2_NODES; ++index){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);

        weights_H1_H2[index] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    weights_H2_O = malloc((size_t)H2_NODES * OUTPUT_NODES *sizeof(float));
    if(weights_H2_O == NULL){
        fprintf(stderr, "Error W_H2_O Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }
    for(int index = 0; index < H2_NODES * OUTPUT_NODES; ++index){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);

        weights_H2_O[index] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    bias_H1 = malloc((size_t)H1_NODES * sizeof(float));
    if(bias_H1 == NULL){
        fprintf(stderr, "Error B_H1 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }
    for(int index = 0; index < H1_NODES; ++index){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);

        bias_H1[index] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    bias_H2 = malloc((size_t)H2_NODES * sizeof(float));
    if(bias_H2 == NULL){
        fprintf(stderr, "Error B_H2 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }
    for(int index = 0; index < H2_NODES; ++index){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);

        bias_H2[index] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    bias_O = malloc((size_t)OUTPUT_NODES * sizeof(float));
    if(bias_O == NULL){
        fprintf(stderr, "Error B_O Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }
    for(int index = 0; index < OUTPUT_NODES; ++index){
        float random_0_1 = (float) rand() / RAND_MAX;
        float scale = sqrt(2.0f / OUTPUT_NODES);

        bias_O[index] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    H1_OUTPUT = malloc((size_t)H1_NODES * sizeof(float));
    if(H1_OUTPUT == NULL){
        fprintf(stderr, "Error O_H1 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }

    H2_OUTPUT = malloc((size_t)H2_NODES * sizeof(float));
    if(H2_OUTPUT == NULL){
        fprintf(stderr, "Error O_H2 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }

    O_OUTPUT = malloc((size_t)OUTPUT_NODES * sizeof(float));
    if(O_OUTPUT == NULL){
        fprintf(stderr, "Error O_O Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }

    error_O = malloc((size_t)OUTPUT_NODES * sizeof(float));
    if(error_O == NULL){
        fprintf(stderr, "Error E_O Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }

    error_H2 = malloc((size_t)H2_NODES * sizeof(float));
    if(error_H2 == NULL){
        fprintf(stderr, "Error E_H2 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }

    error_H1 = malloc((size_t)H1_NODES * sizeof(float));
    if(error_H1 == NULL){
        fprintf(stderr, "Error E_H1 Memory Allocation Error\n");
        terminate_program(1, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
    }

    printf("Training Started!\n");

    float correct_probability = 0;
    int correct = 0;
    int total = 0;
    float current_time = 0;
    float total_time = 0;
    clock_t start_time = 0;
    clock_t end_time = 0;

    // Pre-Calculations for speed up
    int RxC = IMAGE_ROWS * IMAGE_COLS;
    int H2nS = H2_NODES * sizeof(float);
    int H1nS = H1_NODES * sizeof(float);

    for(int EPOCH = 1; EPOCH <= EPOCHS; ++EPOCH){
        correct_probability = 0;
        correct = 0;
        total = 0;

        // Training

        start_time = clock();

        for(int image = 0; image < TRAIN_IMAGE_COUNT; ++image){
            int index = image * RxC;

            // Forward-Propagation

            for(int node_h1 = 0; node_h1 < H1_NODES; ++node_h1){
                H1_OUTPUT[node_h1] = bias_H1[node_h1];
                int ind = 0;
                int limit = node_h1 * INPUT_NODES + INPUT_NODES;
                for(int weight = node_h1 * INPUT_NODES; weight < limit; ++weight){
                    H1_OUTPUT[node_h1] += weights_I_H1[weight] * train_pixels[index + ind++];
                }

                if(H1_OUTPUT[node_h1] < 0) H1_OUTPUT[node_h1] = 0;
            }

            for(int node_h2 = 0; node_h2 < H2_NODES; ++node_h2){
                H2_OUTPUT[node_h2] = bias_H2[node_h2];
                int ind = 0;
                int limit = node_h2 * H1_NODES + H1_NODES;
                for(int weight = node_h2 * H1_NODES; weight < limit; ++weight){
                    H2_OUTPUT[node_h2] += weights_H1_H2[weight] * H1_OUTPUT[ind++];
                }

                if(H2_OUTPUT[node_h2] < 0) H2_OUTPUT[node_h2] = 0;
            }

            float min_output = FLT_MAX;
            float sum = 0;

            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                O_OUTPUT[node_o] = bias_O[node_o];
                int ind = 0;
                int limit = node_o * H2_NODES + H2_NODES;
                for(int weight = node_o * H2_NODES; weight < limit; ++weight){
                    O_OUTPUT[node_o] += weights_H2_O[weight] * H2_OUTPUT[ind++];
                }

                if(O_OUTPUT[node_o] < 0) O_OUTPUT[node_o] = 0;
                if(O_OUTPUT[node_o] != 0 && min_output > O_OUTPUT[node_o]) min_output = O_OUTPUT[node_o];
            }

            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                if(O_OUTPUT[node_o] != 0) O_OUTPUT[node_o] -= min_output;
                O_OUTPUT[node_o] = expf(O_OUTPUT[node_o]);
                sum += O_OUTPUT[node_o];
            }
            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                O_OUTPUT[node_o] /= sum;
            }

            // Backward-Propagation

            uint8_t correct_label = train_labels[image];
            float expected_labels[OUTPUT_NODES] = {0};
            expected_labels[correct_label] = 1;

            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                error_O[node_o] = O_OUTPUT[node_o] - expected_labels[node_o];
            }

            memset(error_H2, 0, H2nS);
            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                int ind = 0;
                int limit = node_o * H2_NODES + H2_NODES;
                for(int weight = node_o * H2_NODES; weight < limit; ++weight){
                    if(H2_OUTPUT[ind] != 0) error_H2[ind] += error_O[node_o] * weights_H2_O[weight];
                    ++ind;
                }
            }

            memset(error_H1, 0, H1nS);
            for(int node_h2 = 0; node_h2 < H2_NODES; ++node_h2){
                int ind = 0;
                int limit = node_h2 * H1_NODES + H1_NODES;
                for(int weight = node_h2 * H1_NODES; weight < limit; ++weight){
                    if(H1_OUTPUT[ind] != 0) error_H1[ind] += error_H2[node_h2] * weights_H1_H2[weight];
                    ++ind;
                }
            }

            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                float leo = LEARNING_RATE * error_O[node_o];
                bias_O[node_o] -= leo;

                int ind = 0;
                int limit = node_o * H2_NODES + H2_NODES;
                for(int weight = node_o * H2_NODES; weight < limit; ++weight){
                    weights_H2_O[weight] -= leo * H2_OUTPUT[ind++];
                }
            }

            for(int node_h2 = 0; node_h2 < H2_NODES; ++node_h2){
                float leh2 = LEARNING_RATE * error_H2[node_h2];
                bias_H2[node_h2] -= leh2;

                int ind = 0;
                int limit = node_h2 * H1_NODES + H1_NODES;
                for(int weight = node_h2 * H1_NODES; weight < limit; ++weight){
                    weights_H1_H2[weight] -= leh2 * H1_OUTPUT[ind++];
                }
            }

            for(int node_h1 = 0; node_h1 < H1_NODES; ++node_h1){
                float leh1 = LEARNING_RATE * error_H1[node_h1];
                bias_H1[node_h1] -= leh1;

                int ind = 0;
                int limit = node_h1 * INPUT_NODES + INPUT_NODES;
                for(int weight = node_h1 * INPUT_NODES; weight < limit; ++weight){
                    weights_I_H1[weight] -= leh1 * train_pixels[index + ind++];
                }
            }
        }

        end_time = clock();
        current_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_time += current_time;

        // Testing

        for(int image = 0; image < TEST_IMAGE_COUNT; ++image){
            int index = image * RxC;

            // Forward-Propagation

            for(int node_h1 = 0; node_h1 < H1_NODES; ++node_h1){
                H1_OUTPUT[node_h1] = bias_H1[node_h1];
                int ind = 0;
                int limit = node_h1 * INPUT_NODES + INPUT_NODES;
                for(int weight = node_h1 * INPUT_NODES; weight < limit; ++weight){
                    H1_OUTPUT[node_h1] += weights_I_H1[weight] * test_pixels[index + ind++];
                }

                if(H1_OUTPUT[node_h1] < 0) H1_OUTPUT[node_h1] = 0;
            }

            for(int node_h2 = 0; node_h2 < H2_NODES; ++node_h2){
                H2_OUTPUT[node_h2] = bias_H2[node_h2];
                int ind = 0;
                int limit = node_h2 * H1_NODES + H1_NODES;
                for(int weight = node_h2 * H1_NODES; weight < limit; ++weight){
                    H2_OUTPUT[node_h2] += weights_H1_H2[weight] * H1_OUTPUT[ind++];
                }

                if(H2_OUTPUT[node_h2] < 0) H2_OUTPUT[node_h2] = 0;
            }

            float min_output = FLT_MAX;
            float sum = 0;

            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                O_OUTPUT[node_o] = bias_O[node_o];
                int ind = 0;
                int limit = node_o * H2_NODES + H2_NODES;
                for(int weight = node_o * H2_NODES; weight < limit; ++weight){
                    O_OUTPUT[node_o] += weights_H2_O[weight] * H2_OUTPUT[ind++];
                }

                if(O_OUTPUT[node_o] < 0) O_OUTPUT[node_o] = 0;
                if(O_OUTPUT[node_o] != 0 && min_output > O_OUTPUT[node_o]) min_output = O_OUTPUT[node_o];
            }

            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                if(O_OUTPUT[node_o] != 0) O_OUTPUT[node_o] -= min_output;
                O_OUTPUT[node_o] = expf(O_OUTPUT[node_o]);
                sum += O_OUTPUT[node_o];
            }
            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                O_OUTPUT[node_o] /= sum;
            }

            uint8_t correct_label = test_labels[image];
            float expected_labels[OUTPUT_NODES] = {0};
            expected_labels[correct_label] = 1;

            int greatest = 1;
            for(int node_o = 0; node_o < OUTPUT_NODES; ++node_o){
                if(node_o != correct_label && O_OUTPUT[correct_label] <= O_OUTPUT[node_o]){
                    greatest = 0;
                    break;
                }
            }
            if(greatest == 1) ++correct;
            ++total;
        }

        correct_probability = (double)correct / total;

        printf("EPOCH %d : %f sec : %f%% accuracy\n", EPOCH, current_time, correct_probability * 100);
    }

    printf("\nTime : %f sec | EPOCHS : %d | Accuracy : %f%%\n", total_time, EPOCHS, correct_probability * 100);

    terminate_program(0, train_pixels, train_labels, test_pixels, test_labels, weights_I_H1, weights_H1_H2, weights_H2_O, bias_H1, bias_H2, bias_O, H1_OUTPUT, H2_OUTPUT, O_OUTPUT, error_O, error_H2, error_H1);
}