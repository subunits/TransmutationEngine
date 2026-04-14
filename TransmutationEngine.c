#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * TRANSMUTATION ENGINE - Refactored for UQGVS Integrity
 * ============================================================================ */

#define CA_WIDTH 256
#define CA_GENERATIONS 64
#define HOLO_SIZE 128
#define ASCII_RAMP " .:-=+*#%@"
#define RAMPLEN 10

typedef struct { double w, x, y, z; } Quaternion;
typedef struct { Quaternion rotation; double magnitude; } GesturePoint;

typedef struct {
    uint8_t cells[CA_WIDTH];
    uint8_t rule;
} CellularState;

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
// MATH & STAGE 1: QUATERNION GESTURES
// ============================================================================

Quaternion note_to_quaternion(double freq, double dur, double amp) {
    double n_freq = freq / 440.0;
    double angle = 2.0 * M_PI * n_freq * dur;
    // Map frequency/amplitude to 3D rotation axis
    double s_freq = sin(n_freq * M_PI);
    double c_freq = cos(n_freq * M_PI);
    double axis_norm = sqrt(s_freq * s_freq + c_freq * c_freq + amp * amp);
    
    double ha = angle / 2.0;
    double sha = sin(ha);
    return (Quaternion){ cos(ha), (s_freq/axis_norm)*sha, (c_freq/axis_norm)*sha, (amp/axis_norm)*sha };
}

GesturePoint* parse_musical_data(const char* data, size_t* out_count) {
    size_t cap = 128, count = 0;
    GesturePoint* pts = malloc(cap * sizeof(GesturePoint));
    const char* line = data;

    while (line && *line) {
        double f, d, a;
        if (sscanf(line, "%lf,%lf,%lf", &f, &d, &a) == 3) {
            if (count >= cap) pts = realloc(pts, (cap *= 2) * sizeof(GesturePoint));
            pts[count].rotation = note_to_quaternion(f, d, a);
            pts[count].magnitude = a;
            count++;
        }
        line = strchr(line, '\n');
        if (line) line++;
    }
    *out_count = count;
    return pts;
}

// ============================================================================
// STAGE 2: HIGH-FIDELITY CA EVOLUTION (Rule 78)
// ============================================================================

void quaternion_to_ca_seed(GesturePoint* gps, size_t count, CellularState* ca) {
    memset(ca->cells, 0, CA_WIDTH);
    for (size_t i = 0; i < count && (i * 4 + 3) < CA_WIDTH; i++) {
        // Map 4D hypersphere to spatial CA zones
        ca->cells[i * 4 + 0] = (uint8_t)((gps[i].rotation.w + 1.0) * 127.5);
        ca->cells[i * 4 + 1] = (uint8_t)((gps[i].rotation.x + 1.0) * 127.5);
        ca->cells[i * 4 + 2] = (uint8_t)((gps[i].rotation.y + 1.0) * 127.5);
        ca->cells[i * 4 + 3] = (uint8_t)((gps[i].rotation.z + 1.0) * 127.5);
    }
    ca->rule = 78; 
}

void ca_step(CellularState* curr, CellularState* next) {
    next->rule = curr->rule;
    for (int i = 0; i < CA_WIDTH; i++) {
        uint8_t n = ((curr->cells[(i-1+CA_WIDTH)%CA_WIDTH] > 127) << 2) |
                    ((curr->cells[i] > 127) << 1) |
                    (curr->cells[(i+1)%CA_WIDTH] > 127);
        next->cells[i] = ((curr->rule >> n) & 1) ? 255 : 0;
    }
}

// ============================================================================
// STAGE 3 & 4: HOLOGRAPHIC PROJECTION & ASCII
// ============================================================================

void generate_pipeline_output(TransmutationPipeline* p) {
    // 1. Evolve CA
    p->ca_states = malloc(CA_GENERATIONS * sizeof(CellularState));
    CellularState init;
    quaternion_to_ca_seed(p->quaternion_seq, p->quat_count, &init);
    p->ca_states[0] = init;
    for (int i = 1; i < CA_GENERATIONS; i++) ca_step(&p->ca_states[i-1], &p->ca_states[i]);

    // 2. Holographic Interference (Intensity I = |Ref + Obj|^2)
    p->holo_pattern = malloc(sizeof(HolographicPattern));
    for (int y = 0; y < HOLO_SIZE; y++) {
        for (int x = 0; x < HOLO_SIZE; x++) {
            double obj = (y < CA_GENERATIONS && x < CA_WIDTH) ? p->ca_states[y].cells[x]/255.0 : 0.0;
            double ref = sin(x * 0.3 + y * 0.1); 
            p->holo_pattern->intensity[y][x] = pow(ref + obj, 2) * 0.25; 
        }
    }

    // 3. ASCII Synthesis
    p->ascii_output = malloc((HOLO_SIZE + 1) * (HOLO_SIZE/2) + 1);
    int pos = 0;
    for (int y = 0; y < HOLO_SIZE/2; y++) {
        for (int x = 0; x < HOLO_SIZE; x++) {
            int idx = (int)(p->holo_pattern->intensity[y*2][x] * (RAMPLEN - 1));
            p->ascii_output[pos++] = ASCII_RAMP[idx > 9 ? 9 : (idx < 0 ? 0 : idx)];
        }
        p->ascii_output[pos++] = '\n';
    }
    p->ascii_output[pos] = '\0';
}

// ============================================================================
// EXECUTION
// ============================================================================

int main() {
    const char* input = "261.63,0.5,0.8\n293.66,0.5,0.7\n329.63,0.5,0.9\n392.00,0.5,0.8\n440.00,0.5,1.0\n";
    TransmutationPipeline p = {0};
    
    printf("[1/4] Ingesting Harmonics...\n");
    p.quaternion_seq = parse_musical_data(input, &p.quat_count);
    
    printf("[2/4] Evolving CA (Rule 78)...\n");
    printf("[3/4] Projecting Holographic interference...\n");
    printf("[4/4] Synthesizing ASCII output...\n\n");
    
    generate_pipeline_output(&p);
    printf("%s\n", p.ascii_output);

    free(p.quaternion_seq); free(p.ca_states); free(p.holo_pattern); free(p.ascii_output);
    return 0;
}
