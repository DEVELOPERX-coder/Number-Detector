#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <cmath>

#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>

#define TOTAL_LAYERS 4
#define HIDDEN_LAYERS 2
#define INPUT_NODES 784
#define H1_NODES 64
#define H2_NODES 32
#define OUTPUT_NODES 10
#define LEARNING_RATE 0.01f
#define EPOCHS 10

#define TRAINING_IMAGE_COUNT 60000
#define TEST_IMAGE_COUNT 10000
#define IMAGE_ROW 28
#define IMAGE_COL 28

#define WINDOW_WIDTH 960
#define WINDOW_HEIGHT 540

uint32_t swap_endian(uint32_t val)
{
    return ((val << 24) & 0xFF000000) |
           ((val << 8) & 0x00FF0000) |
           ((val >> 8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

bool read_MNIST_images(const std::string &filename, std::vector<uint8_t> &pixels)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error : Could Not Open File : " << filename << std::endl;
        return false;
    }

    uint32_t magic_number, num_of_images, num_of_rows, num_of_cols;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    if (magic_number != 2051)
    {
        std::cerr << "Error : Invalid Magic Number : " << magic_number << " (expected 2051)" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&num_of_images), sizeof(num_of_images));
    num_of_images = swap_endian(num_of_images);

    file.read(reinterpret_cast<char *>(&num_of_rows), sizeof(num_of_rows));
    num_of_rows = swap_endian(num_of_rows);

    file.read(reinterpret_cast<char *>(&num_of_cols), sizeof(num_of_cols));
    num_of_cols = swap_endian(num_of_cols);

    pixels.resize((size_t)num_of_images * num_of_rows * num_of_cols);
    file.read(reinterpret_cast<char *>(pixels.data()), pixels.size());

    return true;
}

bool read_MNIST_labels(const std::string &filename, std::vector<uint8_t> &labels)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error : Could Not Open File : " << filename << std::endl;
        return false;
    }

    uint32_t magic_number, num_of_labels;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    if (magic_number != 2049)
    {
        std::cerr << "Error : Invalid Magic Number : " << magic_number << " (expected 2049)" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&num_of_labels), sizeof(num_of_labels));
    num_of_labels = swap_endian(num_of_labels);

    labels.resize(num_of_labels);
    file.read(reinterpret_cast<char *>(labels.data()), labels.size());

    return true;
}

int main(int argc, char *argv[])
{
    std::srand(std::time(0));

    std::vector<uint8_t> train_pixels;
    if (read_MNIST_images("./MNIST/train-images.idx3-ubyte", train_pixels))
    {
        std::cout << "Success Train_Image Bytes : " << train_pixels.size() << std::endl;
    }
    else
    {
        std::cout << "Failed Train Pixels Loading!" << std::endl;
        return 1;
    }

    std::vector<uint8_t> train_labels;
    if (read_MNIST_labels("./MNIST/train-labels.idx1-ubyte", train_labels))
    {
        std::cout << "Success Train_Labels Bytes : " << train_labels.size() << std::endl;
    }
    else
    {
        std::cout << "Failed Train Labels Loading!" << std::endl;
        return 1;
    }

    std::vector<uint8_t> test_pixels;
    if (read_MNIST_images("./MNIST/t10k-images.idx3-ubyte", test_pixels))
    {
        std::cout << "Success Test_Image Bytes : " << test_pixels.size() << std::endl;
    }
    else
    {
        std::cout << "Failed Test Pixels Loading!" << std::endl;
        return 1;
    }

    std::vector<uint8_t> test_labels;
    if (read_MNIST_labels("./MNIST/t10k-labels.idx1-ubyte", test_labels))
    {
        std::cout << "Success Test_Labels Bytes : " << test_labels.size() << std::endl;
    }
    else
    {
        std::cout << "Failed Test Labels Loading!" << std::endl;
        return 1;
    }

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Unable to Initialized SDL : %s", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Number Detector", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    if (!window)
    {
        SDL_Log("Unable to Initialized Window : %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer)
    {
        SDL_Log("Unable to Initialized Renderer : %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::vector<float> weights_I_H1(INPUT_NODES * H1_NODES);
    for (int i = 0; i < weights_I_H1.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / INPUT_NODES);

        weights_I_H1[i] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    std::vector<float> weights_H1_H2(H1_NODES * H2_NODES);
    for (int i = 0; i < weights_H1_H2.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);

        weights_H1_H2[i] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    std::vector<float> weights_H2_O(H2_NODES * OUTPUT_NODES);
    for (int i = 0; i < weights_H2_O.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);

        weights_H2_O[i] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    std::vector<float> bias_H1(H1_NODES);
    for (int i = 0; i < bias_H1.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);

        bias_H1[i] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    std::vector<float> bias_H2(H2_NODES);
    for (int i = 0; i < bias_H2.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);

        bias_H2[i] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    std::vector<float> bias_O(OUTPUT_NODES);
    for (int i = 0; i < bias_O.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / OUTPUT_NODES);

        bias_O[i] = (random_0_1 * 2.0f - 1.0f) * scale;
    }

    bool windowLoopRunning = true;
    SDL_Event e;

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}