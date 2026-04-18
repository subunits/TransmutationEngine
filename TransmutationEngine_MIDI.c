#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * TRANSMUTATION ENGINE - MIDI Refactor
 * ============================================================================ */

#define CA_WIDTH 256
#define CA_GENERATIONS 64
#define HOLO_SIZE 128
#define ASCII_RAMP " .:-=+*#%@"

typedef struct { double w, x, y, z; } Quaternion;

typedef struct {
    Quaternion rotation;
    double timestamp;
    uint8_t midi_note;
    uint8_t velocity;
} GesturePoint;

typedef struct { uint8_t cells[CA_WIDTH]; uint8_t rule; } CellularState;

typedef struct {
    double intensity[HOLO_SIZE][HOLO_SIZE];
} HolographicPattern;

typedef struct {
    GesturePoint* quaternion_seq;
    size_t quat_count;
    CellularState* ca_states;
    HolographicPattern* holo_pattern;
    char* ascii_output;
} TransmutationPipeline;

// ============================================================================
// STAGE 1: MIDI -> QUATERNION (Spherical Mapping)
// ============================================================================

Quaternion midi_to_quaternion(uint8_t note, double dur, uint8_t vel) {
    // Normalize inputs
    double n = note / 127.0;
    double v = vel / 127.0;
    double theta = 2.0 * M_PI * n * dur; 
    
    // Create rotation axis based on pitch and velocity
    double s_theta = sin(theta / 2.0);
    return (Quaternion){
        cos(theta / 2.0), 
        sin(n * M_PI) * s_theta, 
        cos(n * M_PI) * s_theta, 
        v * s_theta
    };
}

GesturePoint* parse_midi_data(const char* data, size_t* out_count) {
    size_t cap = 128, count = 0;
    GesturePoint* pts = malloc(cap * sizeof(GesturePoint));
    const char* line = data;

    while (line && *line) {
        int n, v; double d;
        if (sscanf(line, "%d,%lf,%d", &n, &d, &v) == 3) {
            if (count >= cap) pts = realloc(pts, (cap *= 2) * sizeof(GesturePoint));
            pts[count].rotation = midi_to_quaternion(n, d, v);
            pts[count].timestamp = d;
            pts[count].midi_note = n;
            pts[count].velocity = v;
            count++;
        }
        line = strchr(line, '\n');
        if (line) line++;
    }
    *out_count = count;
    return pts;
}

// ============================================================================
// STAGE 2: QUATERNION -> CA (Rule 110 for Complexity)
// ============================================================================

void ca_step(CellularState* curr, CellularState* next) {
    next->rule = curr->rule;
    for (int i = 0; i < CA_WIDTH; i++) {
        uint8_t n = ((curr->cells[(i - 1 + CA_WIDTH) % CA_WIDTH] > 127) << 2) |
                    ((curr->cells[i] > 127) << 1) |
                    (curr->cells[(i + 1) % CA_WIDTH] > 127);
        next->cells[i] = ((curr->rule >> n) & 1) ? 255 : 0;
    }
}

// ============================================================================
// STAGE 3: HOLOGRAPHIC INTERFERENCE
// ============================================================================

void synthesize_pipeline(TransmutationPipeline* p) {
    // 1. Seed CA using the average rotation of the sequence
    p->ca_states = malloc(CA_GENERATIONS * sizeof(CellularState));
    memset(p->ca_states[0].cells, 0, CA_WIDTH);
    
    for (size_t i = 0; i < p->quat_count && i < CA_WIDTH; i++) {
        // Map W component to cell state
        p->ca_states[0].cells[i] = (uint8_t)((p->quaternion_seq[i].rotation.w + 1.0) * 127.5);
    }
    p->ca_states[0].rule = 110; // Turing-complete rule for MIDI-to-Visual complexity
    
    for (int i = 1; i < CA_GENERATIONS; i++) ca_step(&p->ca_states[i-1], &p->ca_states[i]);

    // 2. Holographic Projection
    p->holo_pattern = malloc(sizeof(HolographicPattern));
    for (int y = 0; y < HOLO_SIZE; y++) {
        // Use MIDI sequence to modulate the reference beam per row
        Quaternion ref_quat = p->quaternion_seq[y % p->quat_count].rotation;
        for (int x = 0; x < HOLO_SIZE; x++) {
            double obj = p->ca_states[y % CA_GENERATIONS].cells[x % CA_WIDTH] / 255.0;
            double ref = sin(x * ref_quat.x + y * ref_quat.y + ref_quat.z);
            p->holo_pattern->intensity[y][x] = pow(ref + obj, 2) * 0.25;
        }
    }

    // 3. ASCII Rendering
    p->ascii_output = malloc((HOLO_SIZE + 1) * (HOLO_SIZE / 2) + 1);
    int pos = 0;
    for (int y = 0; y < HOLO_SIZE / 2; y++) {
        for (int x = 0; x < HOLO_SIZE; x++) {
            int idx = (int)(p->holo_pattern->intensity[y * 2][x] * 9);
            p->ascii_output[pos++] = ASCII_RAMP[idx > 9 ? 9 : (idx < 0 ? 0 : idx)];
        }
        p->ascii_output[pos++] = '\n';
    }
    p->ascii_output[pos] = '\0';
}

int main() {
    const char* midi_input = "60,0.5,100\n64,0.5,90\n67,0.5,110\n72,1.0,127\n";
    TransmutationPipeline p = {0};

    p.quaternion_seq = parse_midi_data(midi_input, &p.quat_count);
    synthesize_pipeline(&p);

    printf("%s\n", p.ascii_output);

    free(p.quaternion_seq); free(p.ca_states); free(p.holo_pattern); free(p.ascii_output);
    return 0;
}
