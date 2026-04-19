#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * TRANSMUTATION ENGINE - Refactored v4.0
 * ============================================================================ */

#define CA_WIDTH 256
#define CA_GENERATIONS 128
#define HOLO_SIZE 128
#define ASCII_RAMP " .:-=+*#%@"

typedef struct { double w, x, y, z; } Quat;

typedef struct {
    Quat rot;
    double freq, dur, amp;
} NotePoint;

typedef struct {
    NotePoint* sequence;
    size_t count;
    uint8_t* ca_grid; // Flattened for cache efficiency
    double (*intensity)[HOLO_SIZE];
    char* ascii;
} Pipeline;

// 1. Math: Spherical Axis-Angle Mapping
Quat note_to_quat(double f, double d, double a) {
    double angle = fmod(f * d, 2.0 * M_PI);
    double s = sin(angle / 2.0);
    // Axis is derived from frequency-amplitude harmonics
    return (Quat){ cos(angle / 2.0), s * sin(f), s * cos(f), s * a };
}

// 2. CA Evolution: Rule 110 (Turing Complete complexity)
void evolve_ca(Pipeline* p) {
    p->ca_grid = calloc(CA_GENERATIONS * CA_WIDTH, sizeof(uint8_t));
    uint8_t rule = (p->count * 31) % 256; // Dynamic rule selection

    // Seed CA with the 'W' (scalar) component of the quaternions
    for (size_t i = 0; i < p->count && i < CA_WIDTH; i++) {
        p->ca_grid[i] = (p->sequence[i].rot.w > 0.0) ? 255 : 0;
    }

    for (int g = 1; g < CA_GENERATIONS; g++) {
        uint8_t* prev = &p->ca_grid[(g - 1) * CA_WIDTH];
        uint8_t* curr = &p->ca_grid[g * CA_WIDTH];
        for (int i = 0; i < CA_WIDTH; i++) {
            uint8_t set = ((prev[(i - 1 + CA_WIDTH) % CA_WIDTH] > 0) << 2) |
                          ((prev[i] > 0) << 1) |
                          (prev[(i + 1) % CA_WIDTH] > 0);
            curr[i] = ((rule >> set) & 1) ? 255 : 0;
        }
    }
}

// 3. Holographic Lens: Quaternions warping the CA pattern
void generate_hologram(Pipeline* p) {
    p->intensity = malloc(sizeof(double[HOLO_SIZE][HOLO_SIZE]));
    for (int y = 0; y < HOLO_SIZE; y++) {
        // Use the Quaternion from the music sequence as a "lens" for this row
        Quat lens = p->sequence[y % p->count].rot;
        for (int x = 0; x < HOLO_SIZE; x++) {
            double obj = p->ca_grid[(y % CA_GENERATIONS) * CA_WIDTH + (x % CA_WIDTH)] / 255.0;
            // Interference beam modulated by Quaternion vector components (x, y, z)
            double ref = sin(x * lens.x + y * lens.y + lens.z);
            double val = pow(ref + obj, 2) * 0.25;
            p->intensity[y][x] = val > 1.0 ? 1.0 : val;
        }
    }
}

// 4. ASCII Synthesis
void render(Pipeline* p) {
    p->ascii = malloc((HOLO_SIZE + 1) * (HOLO_SIZE / 2) + 1);
    int pos = 0;
    for (int y = 0; y < HOLO_SIZE; y += 2) {
        for (int x = 0; x < HOLO_SIZE; x++) {
            int idx = (int)(p->intensity[y][x] * 9);
            p->ascii[pos++] = ASCII_RAMP[idx];
        }
        p->ascii[pos++] = '\n';
    }
    p->ascii[pos] = '\0';
}

int main() {
    Pipeline p = { .count = 8 };
    p.sequence = malloc(p.count * sizeof(NotePoint));
    
    // Simple Harmonic input
    double freqs[] = {261.6, 293.6, 329.6, 349.2, 392.0, 440.0, 493.8, 523.2};
    for(int i=0; i<8; i++) {
        p.sequence[i].rot = note_to_quat(freqs[i], 0.5, 0.8);
    }

    evolve_ca(&p);
    generate_hologram(&p);
    render(&p);

    printf("--- TRANSMUTATION COMPLETE ---\n%s", p.ascii);

    free(p.sequence); free(p.ca_grid); free(p.intensity); free(p.ascii);
    return 0;
}
