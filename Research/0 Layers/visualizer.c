#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple 5x7 bitmap font for 0-9, ., %, P, r, o, b, :, space
// 0-9 are indices 0-9. 
// 10: .
// 11: %
// 12: P
// 13: r
// 14: o
// 15: b
// 16: :
// 17: space
const unsigned char font[18][7] = {
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
    {0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F}, // 2
    {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}, // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
    {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}, // 5
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}, // 9
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}, // . (10)
    {0x11, 0x0A, 0x04, 0x08, 0x14, 0x00, 0x00}, // % (11)
    {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}, // P (12)
    {0x00, 0x0B, 0x0C, 0x10, 0x10, 0x10, 0x10}, // r (13)
    {0x00, 0x0E, 0x11, 0x11, 0x11, 0x11, 0x0E}, // o (14)
    {0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E}, // b (15)
    {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}, // : (16)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}  // space (17)
};

int get_char_index(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c == '.') return 10;
    if (c == '%') return 11;
    if (c == 'P') return 12;
    if (c == 'r') return 13;
    if (c == 'o') return 14;
    if (c == 'b') return 15;
    if (c == ':') return 16;
    if (c == ' ') return 17;
    return 17; // Default space
}

void get_pixel_color(int r_in, int g_in, int b_in, int* r_out, int* g_out, int* b_out) {
    *r_out = r_in; *g_out = g_in; *b_out = b_in;
}

// Color mapping: 
// 0.0 -> Black (0,0,0)
// Positive -> Red -> Yellow (Bright)
// Negative -> Blue -> Cyan (Opposite Bright)
void map_color(double value, double max_abs, int* r, int* g, int* b) {
    if (max_abs == 0) {
        *r = *g = *b = 0; // All black if no range
        return;
    }

    double t = value / max_abs; // Normalized to [-1, 1]

    if (t > 0) {
        // Positive: Black -> Red -> Yellow
        if (t < 0.5) {
            double local_t = t * 2.0;
            *r = (int)(255 * local_t);
            *g = 0;
            *b = 0;
        } else {
            double local_t = (t - 0.5) * 2.0;
            *r = 255;
            *g = (int)(255 * local_t);
            *b = 0;
        }
    } else {
        // Negative: Black -> Blue -> Cyan
        t = -t; // Make positive for calculation
        if (t < 0.5) {
            double local_t = t * 2.0;
            *r = 0;
            *g = 0;
            *b = (int)(255 * local_t);
        } else {
            double local_t = (t - 0.5) * 2.0;
            *r = 0;
            *g = (int)(255 * local_t);
            *b = 255;
        }
    }
}

int main() {
    FILE* file = fopen("data/model.bin", "rb");
    if (!file) {
        perror("Error opening data/model.bin");
        return 1;
    }

    int input_nodes, output_nodes;
    fread(&input_nodes, sizeof(int), 1, file);
    fread(&output_nodes, sizeof(int), 1, file);

    double* weights = (double*)malloc(input_nodes * output_nodes * sizeof(double));
    double* biases = (double*)malloc(output_nodes * sizeof(double));
    double* accuracies = (double*)malloc(output_nodes * sizeof(double));

    fread(weights, sizeof(double), input_nodes * output_nodes, file);
    fread(biases, sizeof(double), output_nodes, file);
    // Try reading accuracies. If it fails (old model file), assume 0.
    size_t read_count = fread(accuracies, sizeof(double), output_nodes, file);
    if (read_count < output_nodes) {
        printf("Warning: Could not read accuracies (model.bin might be old). Defaulting to 0.\n");
        for(int i=0; i<output_nodes; ++i) accuracies[i] = 0.0;
    }
    
    fclose(file);

    int img_rows = 28;
    int img_cols = 28;
    int scale = 10;
    int scaled_rows = img_rows * scale; // 280
    int scaled_cols = img_cols * scale; // 280
    
    int gap = 10;
    int stripe_width = 30;
    int text_height = 40;

    int out_width = scaled_cols + gap + stripe_width; 
    int out_height = scaled_rows + text_height;

    for (int o = 0; o < output_nodes; o++) {
        double max_abs = 0;

        // Apply bias to weights and find max absolute value
        // We calculate max based on weight+bias, same as before, to keep consistency
        double bias = biases[o];
        for (int i = 0; i < input_nodes; i++) {
            int idx = o * input_nodes + i;
            // Note: We modify weights in-place here which affects visualization.
            // But we also use 'bias' for the stripe.
            weights[idx] += bias; 
            
            double abs_val = fabs(weights[idx]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        // Also check bias against max_abs? 
        // If bias is huge, we should probably scale max_abs to include it?
        // But usually weights dominate or are similar scale.
        if (fabs(bias) > max_abs) max_abs = fabs(bias);


        char filename[64];
        sprintf(filename, "data/visual_node_%d.ppm", o);
        FILE* fp = fopen(filename, "w");
        if (!fp) {
            perror("Error creating image file");
            continue;
        }

        fprintf(fp, "P3\n%d %d\n255\n", out_width, out_height);

        char text_buf[32];
        sprintf(text_buf, "Prob: %.1f%%", accuracies[o] * 100);

        for (int y = 0; y < out_height; y++) {
            for (int x = 0; x < out_width; x++) {
                int r = 0, g = 0, b = 0;

                if (y < scaled_rows) {
                    if (x < scaled_cols) {
                        // Weight area
                        int w_x = x / scale;
                        int w_y = y / scale;
                        int weight_idx = o * input_nodes + (w_y * img_cols + w_x);
                        map_color(weights[weight_idx], max_abs, &r, &g, &b);
                    } else if (x < scaled_cols + gap) {
                        // Gap (Black)
                        r = g = b = 0;
                    } else {
                        // Stripe (Color Legend: +Max -> 0 -> -Max)
                        double t = (double)y / (double)(scaled_rows - 1);
                        double val = max_abs * (1.0 - 2.0 * t);
                        map_color(val, max_abs, &r, &g, &b);
                    }
                } else {
                    // Text area (Black background, white text)
                    // Draw text "Prob: XX.X%"
                    // Font scaling: 2x?
                    int text_scale = 2;
                    int char_w = 6 * text_scale; // 5 pixels + 1 spacing
                    int char_h = 8 * text_scale; // 7 pixels + 1 spacing
                    
                    int text_start_x = 10;
                    int text_start_y = scaled_rows + (text_height - char_h) / 2;

                    int relative_x = x - text_start_x;
                    int relative_y = y - text_start_y;

                    if (relative_x >= 0 && relative_y >= 0 && relative_y < char_h) {
                        int char_idx = relative_x / char_w;
                        if (char_idx < 32 && text_buf[char_idx] != '\0') {
                            int char_col = (relative_x % char_w) / text_scale;
                            int char_row = relative_y / text_scale;
                            
                            if (char_col < 5 && char_row < 7) {
                                int font_idx = get_char_index(text_buf[char_idx]);
                                // Check bit
                                // Row 0 is top. 0x1C = 00011100
                                // Bit 0 is left? Usually LSB is right.
                                // Let's assume MSB left: 0x1C = 0001 1100 -> ...XXX..
                                // Actually, let's assume bit 0 (0x01) is right-most column (col 4).
                                // 0x10 is left-most (col 0)? No, 5 bits.
                                // 0x20 would be bit 5.
                                // Let's check '0' row 0: 0x1C = 28 = 11100 binary.
                                // Col 0..4.
                                // If 11100 means col 0,1,2 on? No, usuall 0..4 left to right.
                                // 11100: Bit 4 is 1, Bit 3 is 1, Bit 2 is 1.
                                // If Bit 4 is col 0.
                                // (0x1C >> (4 - char_col)) & 1 ?
                                int bit = (font[font_idx][char_row] >> (4 - char_col)) & 1;
                                if (bit) {
                                    r = g = b = 255;
                                }
                            }
                        }
                    }
                }
                
                fprintf(fp, "%d %d %d ", r, g, b);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);
        printf("Generated %s\n", filename);
    }

    free(weights);
    free(biases);
    free(accuracies);

    return 0;
}
