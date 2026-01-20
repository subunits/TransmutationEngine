#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * TRANSMUTATION ENGINE - OnlineGDB Demo Version
 * 
 * A pipeline for transforming data between representational domains:
 * Musical Notation → Quaternion Space → Cellular Automaton → Holographic → ASCII
 * ============================================================================ */

// ============================================================================
// QUATERNION STRUCTURES
// ============================================================================

typedef struct {
    double w, x, y, z;
} Quaternion;

typedef struct {
    Quaternion rotation;
    double timestamp;
    double magnitude;
} GesturePoint;

// ============================================================================
// CELLULAR AUTOMATON
// ============================================================================

#define CA_WIDTH 256
#define CA_GENERATIONS 64

typedef struct {
    uint8_t cells[CA_WIDTH];
    uint8_t rule;
} CellularState;

// ============================================================================
// HOLOGRAPHIC ENCODING
// ============================================================================

#define HOLO_SIZE 128

typedef struct {
    double intensity[HOLO_SIZE][HOLO_SIZE];
    double phase[HOLO_SIZE][HOLO_SIZE];
    double amplitude[HOLO_SIZE][HOLO_SIZE];
} HolographicPattern;

// ============================================================================
// PIPELINE STATE
// ============================================================================

typedef struct {
    void* source_data;
    size_t source_size;
    GesturePoint* quaternion_seq;
    size_t quat_count;
    CellularState* ca_states;
    size_t ca_gen_count;
    HolographicPattern* holo_pattern;
    char* ascii_output;
    size_t ascii_size;
} TransmutationPipeline;

// ============================================================================
// QUATERNION OPERATIONS
// ============================================================================

Quaternion quat_identity() {
    return (Quaternion){1.0, 0.0, 0.0, 0.0};
}

Quaternion quat_normalize(Quaternion q) {
    double norm = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (norm < 1e-10) return quat_identity();
    return (Quaternion){q.w/norm, q.x/norm, q.y/norm, q.z/norm};
}

Quaternion quat_multiply(Quaternion a, Quaternion b) {
    return (Quaternion){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

Quaternion quat_slerp(Quaternion q1, Quaternion q2, double t) {
    Quaternion result;
    double dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z;
    
    if (fabs(dot) > 0.9995) {
        result.w = q1.w + t * (q2.w - q1.w);
        result.x = q1.x + t * (q2.x - q1.x);
        result.y = q1.y + t * (q2.y - q1.y);
        result.z = q1.z + t * (q2.z - q1.z);
        return quat_normalize(result);
    }
    
    double theta = acos(fabs(dot));
    double sin_theta = sin(theta);
    double w1 = sin((1.0 - t) * theta) / sin_theta;
    double w2 = sin(t * theta) / sin_theta;
    
    if (dot < 0) w2 = -w2;
    
    result.w = w1 * q1.w + w2 * q2.w;
    result.x = w1 * q1.x + w2 * q2.x;
    result.y = w1 * q1.y + w2 * q2.y;
    result.z = w1 * q1.z + w2 * q2.z;
    
    return quat_normalize(result);
}

// ============================================================================
// STAGE 1: PARSE SOURCE → QUATERNION SPACE
// ============================================================================

Quaternion note_to_quaternion(double frequency, double duration, double amplitude) {
    double normalized_freq = frequency / 440.0;
    double angle = 2.0 * M_PI * normalized_freq * duration;
    
    double axis_x = sin(normalized_freq * M_PI);
    double axis_y = cos(normalized_freq * M_PI);
    double axis_z = amplitude;
    
    double axis_norm = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
    if (axis_norm > 1e-10) {
        axis_x /= axis_norm;
        axis_y /= axis_norm;
        axis_z /= axis_norm;
    }
    
    double half_angle = angle / 2.0;
    double sin_half = sin(half_angle);
    
    return (Quaternion){
        cos(half_angle),
        axis_x * sin_half,
        axis_y * sin_half,
        axis_z * sin_half
    };
}

GesturePoint* parse_musical_data(const char* data, size_t size, size_t* out_count) {
    size_t capacity = 1024;
    GesturePoint* points = malloc(capacity * sizeof(GesturePoint));
    size_t count = 0;
    
    const char* ptr = data;
    while (ptr < data + size) {
        double freq, dur, amp;
        if (sscanf(ptr, "%lf,%lf,%lf", &freq, &dur, &amp) == 3) {
            if (count >= capacity) {
                capacity *= 2;
                points = realloc(points, capacity * sizeof(GesturePoint));
            }
            
            points[count].rotation = note_to_quaternion(freq, dur, amp);
            points[count].timestamp = count * dur;
            points[count].magnitude = amp;
            count++;
        }
        
        while (ptr < data + size && *ptr != '\n') ptr++;
        if (ptr < data + size) ptr++;
    }
    
    *out_count = count;
    return points;
}

// ============================================================================
// STAGE 2: QUATERNION → CELLULAR AUTOMATON
// ============================================================================

void quaternion_to_ca_seed(GesturePoint* gestures, size_t count, CellularState* ca) {
    memset(ca->cells, 0, CA_WIDTH);
    
    for (size_t i = 0; i < count && i < CA_WIDTH; i++) {
        double mag = sqrt(
            gestures[i].rotation.x * gestures[i].rotation.x +
            gestures[i].rotation.y * gestures[i].rotation.y +
            gestures[i].rotation.z * gestures[i].rotation.z
        );
        
        ca->cells[i] = (uint8_t)(mag * 255.0);
    }
    
    ca->rule = (uint8_t)(count % 256);
}

void ca_step(CellularState* current, CellularState* next) {
    next->rule = current->rule;
    
    for (int i = 0; i < CA_WIDTH; i++) {
        uint8_t left = current->cells[(i - 1 + CA_WIDTH) % CA_WIDTH];
        uint8_t center = current->cells[i];
        uint8_t right = current->cells[(i + 1) % CA_WIDTH];
        
        uint8_t neighborhood = ((left > 127) << 2) | ((center > 127) << 1) | (right > 127);
        uint8_t new_state = (current->rule >> neighborhood) & 1;
        
        next->cells[i] = new_state ? 255 : 0;
    }
}

CellularState* evolve_ca(CellularState* initial, size_t generations, size_t* out_count) {
    CellularState* states = malloc(generations * sizeof(CellularState));
    
    states[0] = *initial;
    for (size_t i = 1; i < generations; i++) {
        ca_step(&states[i-1], &states[i]);
    }
    
    *out_count = generations;
    return states;
}

// ============================================================================
// STAGE 3: CELLULAR AUTOMATON → HOLOGRAPHIC ENCODING
// ============================================================================

HolographicPattern* ca_to_hologram(CellularState* states, size_t gen_count) {
    HolographicPattern* holo = malloc(sizeof(HolographicPattern));
    memset(holo, 0, sizeof(HolographicPattern));
    
    for (int y = 0; y < HOLO_SIZE && y < (int)gen_count; y++) {
        for (int x = 0; x < HOLO_SIZE && x < CA_WIDTH; x++) {
            double cell_val = states[y].cells[x] / 255.0;
            
            double phase_val = 2.0 * M_PI * cell_val;
            holo->intensity[y][x] = cell_val * cell_val;
            holo->phase[y][x] = phase_val;
            holo->amplitude[y][x] = cell_val;
        }
    }
    
    return holo;
}

// ============================================================================
// STAGE 4: HOLOGRAPHIC → ASCII RENDERING
// ============================================================================

const char ASCII_RAMP[] = " .:-=+*#%@";
#define ASCII_RAMP_LEN 10

char* hologram_to_ascii(HolographicPattern* holo, size_t* out_size) {
    size_t width = HOLO_SIZE;
    size_t height = HOLO_SIZE / 2;
    size_t buffer_size = (width + 1) * height + 1;
    
    char* output = malloc(buffer_size);
    size_t pos = 0;
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            double intensity = holo->intensity[y * 2][x];
            int char_idx = (int)(intensity * (ASCII_RAMP_LEN - 1));
            if (char_idx < 0) char_idx = 0;
            if (char_idx >= ASCII_RAMP_LEN) char_idx = ASCII_RAMP_LEN - 1;
            
            output[pos++] = ASCII_RAMP[char_idx];
        }
        output[pos++] = '\n';
    }
    output[pos] = '\0';
    
    *out_size = pos;
    return output;
}

// ============================================================================
// MAIN PIPELINE EXECUTION
// ============================================================================

TransmutationPipeline* pipeline_create() {
    TransmutationPipeline* p = malloc(sizeof(TransmutationPipeline));
    memset(p, 0, sizeof(TransmutationPipeline));
    return p;
}

void pipeline_destroy(TransmutationPipeline* p) {
    if (!p) return;
    free(p->source_data);
    free(p->quaternion_seq);
    free(p->ca_states);
    free(p->holo_pattern);
    free(p->ascii_output);
    free(p);
}

int pipeline_execute(TransmutationPipeline* p, const char* source_data, size_t source_size) {
    p->quaternion_seq = parse_musical_data(source_data, source_size, &p->quat_count);
    
    CellularState initial_ca;
    quaternion_to_ca_seed(p->quaternion_seq, p->quat_count, &initial_ca);
    p->ca_states = evolve_ca(&initial_ca, CA_GENERATIONS, &p->ca_gen_count);
    
    p->holo_pattern = ca_to_hologram(p->ca_states, p->ca_gen_count);
    p->ascii_output = hologram_to_ascii(p->holo_pattern, &p->ascii_size);
    
    return 0;
}

// ============================================================================
// DEMO PATTERNS
// ============================================================================

typedef struct {
    const char* name;
    const char* description;
    const char* data;
} DemoPattern;

DemoPattern demos[] = {
    {
        "C Major Scale",
        "Simple ascending scale demonstrating basic transmutation",
        "261.63,0.5,0.8\n293.66,0.5,0.7\n329.63,0.5,0.9\n349.23,0.5,0.6\n392.00,0.5,0.8\n440.00,0.5,1.0\n493.88,0.5,0.7\n523.25,1.0,0.9\n"
    },
    {
        "Chromatic Descent",
        "Half-step descending pattern showing higher frequency variation",
        "880.00,0.25,1.0\n830.61,0.25,0.9\n783.99,0.25,0.8\n739.99,0.25,0.7\n698.46,0.25,0.6\n659.25,0.25,0.5\n622.25,0.25,0.4\n587.33,0.25,0.3\n"
    },
    {
        "Harmonic Series",
        "Overtone sequence revealing holographic interference patterns",
        "110.00,0.5,1.0\n220.00,0.5,0.8\n330.00,0.5,0.6\n440.00,0.5,0.5\n550.00,0.5,0.4\n660.00,0.5,0.3\n770.00,0.5,0.2\n880.00,0.5,0.1\n"
    }
};

#define NUM_DEMOS 3

// ============================================================================
// MAIN DEMO
// ============================================================================

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║          TRANSMUTATION ENGINE - OnlineGDB Demo                  ║\n");
    printf("║                                                                  ║\n");
    printf("║  Music → Quaternions → Cellular Automaton → Hologram → ASCII    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    for (int demo_num = 0; demo_num < NUM_DEMOS; demo_num++) {
        DemoPattern* demo = &demos[demo_num];
        
        printf("\n");
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf(" DEMO %d: %s\n", demo_num + 1, demo->name);
        printf(" %s\n", demo->description);
        printf("═══════════════════════════════════════════════════════════════════\n\n");
        
        TransmutationPipeline* pipeline = pipeline_create();
        
        printf("⚙ Processing: Musical data → Quaternion space...\n");
        pipeline_execute(pipeline, demo->data, strlen(demo->data));
        
        printf("⚙ Pipeline complete:\n");
        printf("   • Quaternion gestures: %zu\n", pipeline->quat_count);
        printf("   • Wolfram CA Rule: %d\n", pipeline->ca_states[0].rule);
        printf("   • CA generations: %zu\n", pipeline->ca_gen_count);
        printf("   • Hologram size: %dx%d\n", HOLO_SIZE, HOLO_SIZE);
        printf("   • ASCII output: %zu characters\n\n", pipeline->ascii_size);
        
        printf("───────────────────────────────────────────────────────────────────\n");
        printf("TRANSMUTATION OUTPUT (first 25 lines):\n");
        printf("───────────────────────────────────────────────────────────────────\n");
        
        // Show first 25 lines of output
        char* line = pipeline->ascii_output;
        int lines_shown = 0;
        while (*line && lines_shown < 25) {
            char* next = strchr(line, '\n');
            if (next) {
                printf("%.*s\n", (int)(next - line), line);
                line = next + 1;
                lines_shown++;
            } else {
                printf("%s\n", line);
                break;
            }
        }
        
        printf("───────────────────────────────────────────────────────────────────\n");
        
        // Show first quaternion details
        printf("\n📊 Sample Quaternion (first note):\n");
        printf("   w = %9.6f  (scalar part)\n", pipeline->quaternion_seq[0].rotation.w);
        printf("   x = %9.6f  (i component)\n", pipeline->quaternion_seq[0].rotation.x);
        printf("   y = %9.6f  (j component)\n", pipeline->quaternion_seq[0].rotation.y);
        printf("   z = %9.6f  (k component)\n", pipeline->quaternion_seq[0].rotation.z);
        printf("   timestamp = %.3fs\n", pipeline->quaternion_seq[0].timestamp);
        printf("   magnitude = %.3f\n", pipeline->quaternion_seq[0].magnitude);
        
        // Show CA pattern preview
        printf("\n🔬 Cellular Automaton (first generation, first 64 cells):\n   ");
        for (int i = 0; i < 64; i++) {
            printf("%c", pipeline->ca_states[0].cells[i] > 127 ? '#' : '.');
            if (i == 31) printf("\n   ");
        }
        printf("\n");
        
        pipeline_destroy(pipeline);
    }
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    ALL DEMOS COMPLETE                            ║\n");
    printf("║                                                                  ║\n");
    printf("║  Three musical patterns successfully transmuted through:        ║\n");
    printf("║  • Quaternion gesture space (4D rotations)                      ║\n");
    printf("║  • Wolfram cellular automaton evolution                         ║\n");
    printf("║  • Holographic interference encoding                            ║\n");
    printf("║  • ASCII visual rendering                                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}