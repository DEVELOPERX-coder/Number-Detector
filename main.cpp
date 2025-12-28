#include <iostream> // cout, cerr
#include <cstdint>  // uint32_t
#include <fstream>  // file
#include <vector>   // vector
#include <ctime>    // time()
#include <cstdlib>  // rand()
#include <cmath>    // exp()

#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>

#define H_LAYERS 2
#define I_NODES 784
#define H1_NODES 128
#define H2_NODES 64
#define O_NODES 10

#define WINDOW_WIDTH 960
#define WINDOW_HEIGHT 540

#define INPUT_STARTING_X 10
#define INPUT_STARTING_Y WINDOW_HEIGHT / 2

#define SCALE 10
#define ROWS 28
#define COLS 28

#define H1_STARTING_X INPUT_STARTING_X + COLS *SCALE + 10
#define H1_STARTING_Y 10
#define H1_NODE_HEIGHT (WINDOW_HEIGHT - 20) / H1_NODES
#define H1_NODE_WIDTH SCALE

#define H2_STARTING_X H1_STARTING_X + 20
#define H2_STARTING_Y 10
#define H2_NODE_HEIGHT (WINDOW_HEIGHT - 20) / H2_NODES
#define H2_NODE_WIDTH SCALE

#define O_STARTING_X H2_STARTING_X + 20
#define O_STARTING_Y 10
#define O_NODE_HEIGHT (WINDOW_HEIGHT - 20) / O_NODES
#define O_NODE_WIDTH SCALE

uint32_t swap_endian(uint32_t val)
{
    return ((val << 24) & 0xFF000000) |
           ((val << 8) & 0x00FF0000) |
           ((val >> 8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

bool read_mnist_train_images(const std::string &filename, std::vector<uint8_t> &pixels, int &rows, int &cols)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file" << filename << std::endl;
        return false;
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051)
    {
        std::cerr << "Error: Invalic Magic Number " << magic_number << " (expected 2051)" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    num_images = swap_endian(num_images);

    file.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    num_rows = swap_endian(num_rows);

    file.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));
    num_cols = swap_endian(num_cols);

    rows = num_rows;
    cols = num_cols;

    std::cout << "Header Info: " << std::endl;
    std::cout << "Magic: " << magic_number << std::endl;
    std::cout << "Images: " << num_images << std::endl;
    std::cout << "Rows: " << num_rows << std::endl;
    std::cout << "Cols: " << num_cols << std::endl;

    pixels.resize(num_images * num_rows * num_cols);

    file.read(reinterpret_cast<char *>(pixels.data()), pixels.size());

    return true;
}

bool read_mnist_train_images_label(const std::string &filename, std::vector<uint8_t> &train_image_labels)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file" << filename << std::endl;
        return false;
    }

    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);

    if (magic_number != 2049)
    {
        std::cerr << "Error: Invalic Magic Number " << magic_number << " (expected 2049)" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
    num_labels = swap_endian(num_labels);

    std::cout << "Header Info: " << std::endl;
    std::cout << "Magic: " << magic_number << std::endl;
    std::cout << "Labels: " << num_labels << std::endl;

    train_image_labels.resize(num_labels);

    file.read(reinterpret_cast<char *>(train_image_labels.data()), train_image_labels.size());

    return true;
}

void DrawCircle(SDL_Renderer *renderer, int cx, int cy, int diameter,
                uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    int radius = diameter / 2;
    SDL_SetRenderDrawColor(renderer, r, g, b, a);

    int x = radius;
    int y = 0;
    int decision = 1 - x;

    while (y <= x)
    {
        // 8-way symmetry
        // SDL_RenderPoint(renderer, cx + x, cy + y);
        // SDL_RenderPoint(renderer, cx + y, cy + x);
        // SDL_RenderPoint(renderer, cx - y, cy + x);
        // SDL_RenderPoint(renderer, cx - x, cy + y);
        // SDL_RenderPoint(renderer, cx - x, cy - y);
        // SDL_RenderPoint(renderer, cx - y, cy - x);
        // SDL_RenderPoint(renderer, cx + y, cy - x);
        // SDL_RenderPoint(renderer, cx + x, cy - y);

        // Draw horizontal spans
        SDL_RenderLine(renderer, cx - x, cy + y, cx + x, cy + y);
        SDL_RenderLine(renderer, cx - x, cy - y, cx + x, cy - y);
        SDL_RenderLine(renderer, cx - y, cy + x, cx + y, cy + x);
        SDL_RenderLine(renderer, cx - y, cy - x, cx + y, cy - x);

        y++;

        if (decision <= 0)
        {
            decision += 2 * y + 1;
        }
        else
        {
            x--;
            decision += 2 * (y - x) + 1;
        }
    }
}

// void DrawCircle(SDL_Renderer *renderer, float starting_x, float starting_y, int diameter, int8_t r, int8_t g, int8_t b, int8_t t)
// {
//     SDL_SetRenderDrawColor(renderer, r, g, b, t);
//     // Diameter can't be 0
//     if (diameter % 2 == 0)
//         ++diameter;

//     for (int i = (diameter >> 1); i >= 0; --i)
//     {
//         for (int j = 0; j < (diameter >> 1) - i + 1; ++j)
//         {
//             SDL_RenderPoint(renderer, starting_x - j, starting_y - i);
//         }
//         for (int j = 1; j <= (diameter >> 1) - i; ++j)
//         {
//             SDL_RenderPoint(renderer, starting_x + j, starting_y - i);
//         }
//     }

//     for (int i = 1; i <= (diameter >> 1); ++i)
//     {
//         for (int j = 0; j <= (diameter >> 1) - i; ++j)
//         {
//             SDL_RenderPoint(renderer, starting_x - j, starting_y + i);
//         }
//         for (int j = 1; j <= (diameter >> 1) - i; ++j)
//         {
//             SDL_RenderPoint(renderer, starting_x + j, starting_y + i);
//         }
//     }
// }

int main(int argc, char *argv[])
{
    std::vector<std::vector<int8_t>> colorMap(256, std::vector<int8_t>(3));

    for (int i = 0; i < 256; i++)
    {
        float t = i / 255.0f;

        float r, g, b;

        if (t < 0.25f)
        { // blue → cyan
            float k = t / 0.25f;
            r = 0;
            g = k * 255;
            b = 255;
        }
        else if (t < 0.50f)
        { // cyan → green
            float k = (t - 0.25f) / 0.25f;
            r = 0;
            g = 255;
            b = (1 - k) * 255;
        }
        else if (t < 0.75f)
        { // green → yellow
            float k = (t - 0.50f) / 0.25f;
            r = k * 255;
            g = 255;
            b = 0;
        }
        else
        { // yellow → red
            float k = (t - 0.75f) / 0.25f;
            r = 255;
            g = (1 - k) * 255;
            b = 0;
        }

        colorMap[i][0] = (int8_t)r;
        colorMap[i][1] = (int8_t)g;
        colorMap[i][2] = (int8_t)b;
    }

    std::srand(std::time(0));

    std::vector<uint8_t> pixels;
    int rows = 0;
    int cols = 0;

    if (read_mnist_train_images("./MNIST/train-images.idx3-ubyte", pixels, rows, cols))
    {
        std::cout << "Data loaded successfully! Total bytes: " << pixels.size() << std::endl;
    }
    else
    {
        std::cout << "Failed to load data." << std::endl;
        return 1;
    }

    std::vector<uint8_t> train_image_labels;

    if (read_mnist_train_images_label("./MNIST/train-labels.idx1-ubyte", train_image_labels))
    {
        std::cout << "Data loaded successfully! Total bytes: " << train_image_labels.size() << std::endl;
    }
    else
    {
        std::cout << "Failed to load data." << std::endl;
        return 1;
    }

    // -----

    std::vector<uint8_t> pixels2;
    int rows2 = 0;
    int cols2 = 0;

    if (read_mnist_train_images("./MNIST/t10k-images.idx3-ubyte", pixels2, rows2, cols2))
    {
        std::cout << "Data loaded successfully! Total bytes: " << pixels2.size() << std::endl;
    }
    else
    {
        std::cout << "Failed to load data." << std::endl;
        return 1;
    }

    std::vector<uint8_t> train_image_labels2;

    if (read_mnist_train_images_label("./MNIST/t10k-labels.idx1-ubyte", train_image_labels2))
    {
        std::cout << "Data loaded successfully! Total bytes: " << train_image_labels2.size() << std::endl;
    }
    else
    {
        std::cout << "Failed to load data." << std::endl;
        return 1;
    }

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Unable to Initialized SDL : %s", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Number Detector NN Visualizer", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
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

    bool running = true;
    SDL_Event e;
    int current_image_idx = 0;

    std::vector<float> weights_IH1(I_NODES * H1_NODES);
    std::vector<float> weights_H1H2(H1_NODES * H2_NODES);
    std::vector<float> weights_H2O(H2_NODES * O_NODES);

    for (int i = 0; i < weights_IH1.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / I_NODES);
        weights_IH1[i] = (random_0_1 * 2.0f) - 1.0f;
        weights_IH1[i] *= scale;
    }

    for (int i = 0; i < weights_H1H2.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);
        weights_H1H2[i] = (random_0_1 * 2.0f) - 1.0f;
        weights_H1H2[i] *= scale;
    }

    for (int i = 0; i < weights_H2O.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);
        weights_H2O[i] = (random_0_1 * 2.0f) - 1.0f;
        weights_H2O[i] *= scale;
    }

    std::vector<float> H1_bias(H1_NODES);
    std::vector<float> H2_bias(H2_NODES);
    std::vector<float> O_bias(O_NODES);

    for (int i = 0; i < H1_bias.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H1_NODES);
        H1_bias[i] = (random_0_1 * 2.0f) - 1.0f;
        H1_bias[i] *= scale;
    }

    for (int i = 0; i < H2_bias.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / H2_NODES);
        H2_bias[i] = (random_0_1 * 2.0f) - 1.0f;
        H2_bias[i] *= scale;
    }

    for (int i = 0; i < O_bias.size(); ++i)
    {
        float random_0_1 = (float)std::rand() / RAND_MAX;
        float scale = sqrt(2.0f / O_NODES);
        O_bias[i] = (random_0_1 * 2.0f) - 1.0f;
        O_bias[i] *= scale;
    }

    int index1 = 0;
    int index2 = 0;

    float correct_probability;
    int correct = 0;
    int total = 0;

    bool training = true;

    while (running)
    {
        while (SDL_PollEvent(&e) != 0)
        {
            if (e.type == SDL_EVENT_QUIT)
            {
                running = false;
            }
            if (e.type == SDL_EVENT_KEY_DOWN)
            {
                if (e.key.key == SDLK_Q)
                {
                    running = false;
                }
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        if (training == true)
        {
            for (int row = 0; row < rows; ++row)
            {
                for (int col = 0; col < cols; ++col)
                {
                    int index = index1 + (row * cols) + col;
                    uint8_t pixel_val = pixels[index];

                    SDL_FRect pixelRect;
                    pixelRect.x = INPUT_STARTING_X + (col * SCALE);
                    pixelRect.y = INPUT_STARTING_Y + (row * SCALE) - (rows / 2 * SCALE);
                    pixelRect.w = SCALE;
                    pixelRect.h = SCALE;

                    SDL_SetRenderDrawColor(renderer, pixel_val, pixel_val, pixel_val, 255);
                    SDL_RenderFillRect(renderer, &pixelRect);
                }
            }
        }
        else
        {
            for (int row = 0; row < rows2; ++row)
            {
                for (int col = 0; col < cols2; ++col)
                {
                    int index = index1 + (row * cols2) + col;
                    uint8_t pixel_val = pixels2[index];

                    SDL_FRect pixelRect;
                    pixelRect.x = INPUT_STARTING_X + (col * SCALE);
                    pixelRect.y = INPUT_STARTING_Y + (row * SCALE) - (rows2 / 2 * SCALE);
                    pixelRect.w = SCALE;
                    pixelRect.h = SCALE;

                    SDL_SetRenderDrawColor(renderer, pixel_val, pixel_val, pixel_val, 255);
                    SDL_RenderFillRect(renderer, &pixelRect);
                }
            }
        }

        std::vector<float> H1_output(H1_NODES, 0);

        if (training == true)
        {
            for (int i = 0; i < H1_NODES; ++i)
            {
                int ind = 0;
                for (int j = i * I_NODES; j < i * I_NODES + I_NODES; ++j)
                {
                    H1_output[i] += weights_IH1[j] * (float)(pixels[index1 + ind++]) / 255;
                }
                H1_output[i] += H1_bias[i];
                // ReLu
                if (H1_output[i] <= 0)
                    H1_output[i] = 0;
            }
        }
        else
        {
            for (int i = 0; i < H1_NODES; ++i)
            {
                int ind = 0;
                for (int j = i * I_NODES; j < i * I_NODES + I_NODES; ++j)
                {
                    H1_output[i] += weights_IH1[j] * (float)(pixels2[index1 + ind++]) / 255;
                }
                H1_output[i] += H1_bias[i];
                // ReLu
                if (H1_output[i] <= 0)
                    H1_output[i] = 0;
            }
        }

        std::vector<float> H2_output(H2_NODES, 0);

        for (int i = 0; i < H2_NODES; ++i)
        {
            int ind = 0;
            for (int j = i * H1_NODES; j < i * H1_NODES + H1_NODES; ++j)
            {
                H2_output[i] += weights_H1H2[j] * H1_output[ind++];
            }
            H2_output[i] += H2_bias[i];
            // ReLu
            if (H2_output[i] <= 0)
                H2_output[i] = 0;
        }

        std::vector<float> O_output(O_NODES, 0);

        for (int i = 0; i < O_NODES; ++i)
        {
            int ind = 0;
            for (int j = i * H2_NODES; j < i * H2_NODES + H2_NODES; ++j)
            {
                O_output[i] += weights_H2O[j] * H2_output[ind++];
            }
            O_output[i] += O_bias[i];
        }

        float sum = 0;
        float maxValue = 0;
        for (int i = 0; i < O_NODES; ++i)
        {
            if (O_output[i] > maxValue)
                maxValue = O_output[i];
        }
        for (int i = 0; i < O_NODES; ++i)
        {
            sum += exp(O_output[i] - maxValue);
        }
        for (int i = 0; i < O_NODES; ++i)
        {
            O_output[i] = exp(O_output[i] - maxValue) / sum;
        }

        for (int i = 0; i < H1_NODES; ++i)
        {
            SDL_FRect pixelRect;
            pixelRect.x = H1_STARTING_X;
            pixelRect.y = H1_STARTING_Y + (i * H1_NODE_HEIGHT);
            pixelRect.w = SCALE;
            pixelRect.h = H1_NODE_HEIGHT;

            // SDL_SetRenderDrawColor(renderer, H1_output[i] * 255, H1_output[i] * 255, H1_output[i] * 255, 255);
            // SDL_RenderFillRect(renderer, &pixelRect);

            uint8_t ind = (int)abs(H1_output[i] * 255);
            if (ind < 0)
                ind *= -1;

            DrawCircle(renderer, H1_STARTING_X, H1_STARTING_Y + (i * H1_NODE_HEIGHT), H1_NODE_HEIGHT, colorMap[ind][0], colorMap[ind][1], colorMap[ind][2], 255);
        }

        for (int i = 0; i < H2_NODES; ++i)
        {
            SDL_FRect pixelRect;
            pixelRect.x = H2_STARTING_X;
            pixelRect.y = H2_STARTING_Y + (i * H2_NODE_HEIGHT);
            pixelRect.w = SCALE;
            pixelRect.h = H2_NODE_HEIGHT;

            // SDL_SetRenderDrawColor(renderer, H2_output[i] * 255, H2_output[i] * 255, H2_output[i] * 255, 255);
            // SDL_RenderFillRect(renderer, &pixelRect);

            uint8_t ind = (int)abs(H2_output[i] * 255);
            if (ind < 0)
                ind *= -1;

            DrawCircle(renderer, H2_STARTING_X, H2_STARTING_Y + (i * H2_NODE_HEIGHT), H2_NODE_HEIGHT, colorMap[ind][0], colorMap[ind][1], colorMap[ind][2], 255);
        }

        for (int i = 0; i < O_NODES; ++i)
        {
            SDL_FRect pixelRect;
            pixelRect.x = O_STARTING_X;
            pixelRect.y = O_STARTING_Y + (i * O_NODE_HEIGHT);
            pixelRect.w = SCALE;
            pixelRect.h = O_NODE_HEIGHT;

            // SDL_SetRenderDrawColor(renderer, O_output[i] * 255, O_output[i] * 255, O_output[i] * 255, 255);
            // SDL_RenderFillRect(renderer, &pixelRect);

            uint8_t ind = (int)abs(O_output[i] * 255);
            if (ind < 0)
                ind *= -1;

            DrawCircle(renderer, O_STARTING_X, O_STARTING_Y + (i * O_NODE_HEIGHT), O_NODE_HEIGHT / 2, colorMap[ind][0], colorMap[ind][1], colorMap[ind][2], 255);

            SDL_FRect barRect = {O_STARTING_X + SCALE + 20, O_STARTING_Y + (float)(i * O_NODE_HEIGHT), O_output[i] * 150, O_NODE_HEIGHT};
            SDL_SetRenderDrawColor(renderer, 173, 255, 47, 255);
            SDL_RenderFillRect(renderer, &barRect);
        }

        SDL_RenderPresent(renderer);

        // backpropagation

        if (training == true)
        {
            float learning_rate = 0.01;
            uint8_t correct_label = train_image_labels[index2];
            std::vector<float> expected_labels(O_NODES, 0);
            expected_labels[(int)correct_label] = 1;

            if (O_output[correct_label] > 0.50f)
                ++correct;

            std::vector<float> error(O_NODES, 0);
            for (int i = 0; i < O_NODES; ++i)
            {
                error[i] = O_output[i] - expected_labels[i];
            }

            std::vector<float> h2_error(H2_NODES, 0);
            for (int i = 0; i < O_NODES; ++i)
            {
                int ind = 0;
                for (int j = i * H2_NODES; j < i * H2_NODES + H2_NODES; ++j)
                {
                    h2_error[ind] += error[i] * weights_H2O[j];
                    ++ind;
                }
            }
            for (int i = 0; i < H2_NODES; ++i)
            {
                if (H2_output[i] == 0)
                    h2_error[i] = 0;
            }

            for (int i = 0; i < O_NODES; ++i)
            {
                O_bias[i] = O_bias[i] - learning_rate * error[i];
            }

            for (int i = 0; i < O_NODES; ++i)
            {
                int ind = 0;
                for (int j = i * H2_NODES; j < i * H2_NODES + H2_NODES; ++j)
                {
                    weights_H2O[j] = weights_H2O[j] - learning_rate * error[i] * H2_output[ind++];
                }
            }

            std::vector<float> h1_error(H1_NODES, 0);
            for (int i = 0; i < H2_NODES; ++i)
            {
                int ind = 0;
                for (int j = i * H1_NODES; j < i * H1_NODES + H1_NODES; ++j)
                {
                    h1_error[ind++] += h2_error[i] * weights_H1H2[j];
                }
            }
            for (int i = 0; i < H1_NODES; ++i)
            {
                if (H1_output[i] == 0)
                    h1_error[i] = 0;
            }

            for (int i = 0; i < H2_NODES; ++i)
            {
                H2_bias[i] = H2_bias[i] - learning_rate * h2_error[i];
            }

            for (int i = 0; i < H2_NODES; ++i)
            {
                int ind = 0;
                for (int j = i * H1_NODES; j < i * H1_NODES + H1_NODES; ++j)
                {
                    weights_H1H2[j] = weights_H1H2[j] - learning_rate * h2_error[i] * H1_output[ind++];
                }
            }

            for (int i = 0; i < H1_NODES; ++i)
            {
                H1_bias[i] = H1_bias[i] - learning_rate * h1_error[i];
            }

            for (int i = 0; i < H1_NODES; ++i)
            {
                int ind = 0;
                for (int j = i * I_NODES; j < i * I_NODES + I_NODES; ++j)
                {
                    weights_IH1[j] = weights_IH1[j] - learning_rate * h1_error[i] * (float)pixels[index1 + ind] / 255;
                    ++ind;
                }
            }

            index1 += rows * cols;
            ++index2;

            ++total;
            if (total % 5000 == 0)
            {
                correct_probability = (float)correct / total;
                std::cout << correct_probability << std::endl;
            }
            if (index1 == 60000 * rows * cols)
            {
                training = false;
                total = 1;
                correct = 1;
                index1 = 0;
                index2 = 0;
                std::cout << "Training is completed : now running test data" << std::endl;
            }
        }
        else
        {
            float learning_rate = 0.01;
            uint8_t correct_label = train_image_labels2[index2];
            std::vector<float> expected_labels(O_NODES, 0);
            expected_labels[(int)correct_label] = 1;

            if (O_output[correct_label] > 0.50f)
                ++correct;

            index1 += rows2 * cols2;
            ++index2;

            ++total;

            if (total % 5000 == 0)
            {
                correct_probability = (float)correct / total;
                std::cout << correct_probability << std::endl;
            }
            if (index1 == train_image_labels2.size() * rows2 * cols2)
                running = false;
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}