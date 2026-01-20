#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * TRANSMUTATION ENGINE - Core Architecture (VERBOSE MODE)
 * 
 * A pipeline for transforming data between representational domains:
 * Musical Notation → Quaternion Space → Cellular Automaton → Holographic → ASCII
 * ============================================================================ */

// ============================================================================
// QUATERNION STRUCTURES (from your UQGVS work)
// ============================================================================

typedef struct {
    double w, x, y, z;
} Quaternion;

typedef struct {
    Quaternion rotation;
    double timestamp;
    double magnitude;  // For musical amplitude/dynamics
} GesturePoint;

// ============================================================================
// CELLULAR AUTOMATON (from your FDCA-Simulator)
// ============================================================================

#define CA_WIDTH 256
#define CA_GENERATIONS 64

typedef struct {
    uint8_t cells[CA_WIDTH];
    uint8_t rule;  // Wolfram rule number (0-255)
} CellularState;

// ============================================================================
// HOLOGRAPHIC ENCODING (from your Body-Holographic work)
// ============================================================================

#define HOLO_SIZE 128

typedef struct {
    double intensity[HOLO_SIZE][HOLO_SIZE];
    double phase[HOLO_SIZE][HOLO_SIZE];
    double amplitude[HOLO_SIZE][HOLO_SIZE];
} HolographicPattern;

// ============================================================================
// PIPELINE STATE - The Transmutation Flow
// ============================================================================

typedef struct {
    // Source domain
    void* source_data;
    size_t source_size;
    
    // Intermediate representations
    GesturePoint* quaternion_seq;
    size_t quat_count;
    
    CellularState* ca_states;
    size_t ca_gen_count;
    
    HolographicPattern* holo_pattern;
    
    // Final output
    char* ascii_output;
    size_t ascii_size;
    
} TransmutationPipeline;

// ============================================================================
// QUATERNION OPERATIONS
// ============================================================================

Quaternion quat_identity() {
    printf("    [quat] Creating identity quaternion (1,0,0,0)\n");
    return (Quaternion){1.0, 0.0, 0.0, 0.0};
}

Quaternion quat_normalize(Quaternion q) {
    double norm = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    printf("    [quat] Normalizing quaternion: norm=%.6f\n", norm);
    
    if (norm < 1e-10) {
        printf("    [quat] WARNING: Near-zero norm, returning identity\n");
        return quat_identity();
    }
    
    Quaternion result = {q.w/norm, q.x/norm, q.y/norm, q.z/norm};
    printf("    [quat] Normalized: (%.3f, %.3f, %.3f, %.3f)\n", 
           result.w, result.x, result.y, result.z);
    return result;
}

Quaternion quat_multiply(Quaternion a, Quaternion b) {
    printf("    [quat] Multiplying quaternions\n");
    printf("      a = (%.3f, %.3f, %.3f, %.3f)\n", a.w, a.x, a.y, a.z);
    printf("      b = (%.3f, %.3f, %.3f, %.3f)\n", b.w, b.x, b.y, b.z);
    
    Quaternion result = {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
    
    printf("      result = (%.3f, %.3f, %.3f, %.3f)\n", 
           result.w, result.x, result.y, result.z);
    return result;
}

// SLERP interpolation (from your ascii-camera-preview)
Quaternion quat_slerp(Quaternion q1, Quaternion q2, double t) {
    printf("    [quat] SLERP interpolation at t=%.3f\n", t);
    
    Quaternion result;
    double dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z;
    printf("      dot product = %.6f\n", dot);
    
    // If quaternions are very close, use linear interpolation
    if (fabs(dot) > 0.9995) {
        printf("      Using linear interpolation (quaternions very close)\n");
        result.w = q1.w + t * (q2.w - q1.w);
        result.x = q1.x + t * (q2.x - q1.x);
        result.y = q1.y + t * (q2.y - q1.y);
        result.z = q1.z + t * (q2.z - q1.z);
        return quat_normalize(result);
    }
    
    double theta = acos(fabs(dot));
    double sin_theta = sin(theta);
    printf("      theta = %.6f rad, sin(theta) = %.6f\n", theta, sin_theta);
    
    double w1 = sin((1.0 - t) * theta) / sin_theta;
    double w2 = sin(t * theta) / sin_theta;
    
    if (dot < 0) {
        printf("      Negating w2 (dot < 0)\n");
        w2 = -w2;
    }
    
    printf("      weights: w1=%.6f, w2=%.6f\n", w1, w2);
    
    result.w = w1 * q1.w + w2 * q2.w;
    result.x = w1 * q1.x + w2 * q2.x;
    result.y = w1 * q1.y + w2 * q2.y;
    result.z = w1 * q1.z + w2 * q2.z;
    
    return quat_normalize(result);
}

// ============================================================================
// STAGE 1: PARSE SOURCE → QUATERNION SPACE
// ============================================================================

// Musical note to quaternion (frequency → rotation in 3D space)
Quaternion note_to_quaternion(double frequency, double duration, double amplitude) {
    printf("  [note→quat] Converting note: freq=%.2fHz, dur=%.3fs, amp=%.2f\n", 
           frequency, duration, amplitude);
    
    // Map frequency to rotation axis
    double normalized_freq = frequency / 440.0;  // Normalize to A440
    double angle = 2.0 * M_PI * normalized_freq * duration;
    printf("    normalized_freq = %.6f, rotation_angle = %.6f rad\n", 
           normalized_freq, angle);
    
    // Use amplitude to modulate axis
    double axis_x = sin(normalized_freq * M_PI);
    double axis_y = cos(normalized_freq * M_PI);
    double axis_z = amplitude;
    printf("    rotation_axis (unnormalized) = (%.6f, %.6f, %.6f)\n", 
           axis_x, axis_y, axis_z);
    
    // Normalize axis
    double axis_norm = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
    if (axis_norm > 1e-10) {
        axis_x /= axis_norm;
        axis_y /= axis_norm;
        axis_z /= axis_norm;
        printf("    rotation_axis (normalized) = (%.6f, %.6f, %.6f)\n", 
               axis_x, axis_y, axis_z);
    }
    
    // Create quaternion from axis-angle
    double half_angle = angle / 2.0;
    double sin_half = sin(half_angle);
    
    Quaternion result = {
        cos(half_angle),
        axis_x * sin_half,
        axis_y * sin_half,
        axis_z * sin_half
    };
    
    printf("    resulting quaternion = (%.6f, %.6f, %.6f, %.6f)\n", 
           result.w, result.x, result.y, result.z);
    
    return result;
}

// Simplified musical data parser (extend for Sibelius format)
GesturePoint* parse_musical_data(const char* data, size_t size, size_t* out_count) {
    printf("  [parser] Parsing musical data (%zu bytes)\n", size);
    printf("  [parser] Input format: freq,duration,amplitude\\n\n");
    
    size_t capacity = 1024;
    GesturePoint* points = malloc(capacity * sizeof(GesturePoint));
    size_t count = 0;
    
    const char* ptr = data;
    int line_num = 1;
    
    while (ptr < data + size) {
        double freq, dur, amp;
        if (sscanf(ptr, "%lf,%lf,%lf", &freq, &dur, &amp) == 3) {
            printf("  [parser] Line %d: parsed note %.2fHz\n", line_num, freq);
            
            if (count >= capacity) {
                capacity *= 2;
                printf("  [parser] Expanding capacity to %zu\n", capacity);
                points = realloc(points, capacity * sizeof(GesturePoint));
            }
            
            points[count].rotation = note_to_quaternion(freq, dur, amp);
            points[count].timestamp = count * dur;
            points[count].magnitude = amp;
            
            printf("  [parser] GesturePoint #%zu: timestamp=%.3fs, magnitude=%.2f\n\n", 
                   count, points[count].timestamp, points[count].magnitude);
            
            count++;
        }
        
        // Move to next line
        while (ptr < data + size && *ptr != '\n') ptr++;
        if (ptr < data + size) ptr++;
        line_num++;
    }
    
    printf("  [parser] Total parsed: %zu gesture points\n\n", count);
    *out_count = count;
    return points;
}

// ============================================================================
// STAGE 2: QUATERNION → CELLULAR AUTOMATON
// ============================================================================

// Encode quaternion sequence into CA initial state
void quaternion_to_ca_seed(GesturePoint* gestures, size_t count, CellularState* ca) {
    printf("  [quat→ca] Encoding %zu quaternions into CA seed\n", count);
    
    memset(ca->cells, 0, CA_WIDTH);
    
    // Encode quaternion data into bit pattern
    for (size_t i = 0; i < count && i < CA_WIDTH; i++) {
        // Map quaternion magnitude to cell state
        double mag = sqrt(
            gestures[i].rotation.x * gestures[i].rotation.x +
            gestures[i].rotation.y * gestures[i].rotation.y +
            gestures[i].rotation.z * gestures[i].rotation.z
        );
        
        ca->cells[i] = (uint8_t)(mag * 255.0);
        
        printf("    Cell[%zu]: quaternion_mag=%.6f → cell_value=%d\n", 
               i, mag, ca->cells[i]);
    }
    
    // Use gesture count to influence rule selection
    ca->rule = (uint8_t)(count % 256);
    printf("  [quat→ca] Selected Wolfram rule: %d (based on count %zu)\n", 
           ca->rule, count);
    printf("  [quat→ca] Rule %d binary: ", ca->rule);
    for (int i = 7; i >= 0; i--) {
        printf("%d", (ca->rule >> i) & 1);
    }
    printf("\n\n");
}

// Evolve cellular automaton (Wolfram rules)
void ca_step(CellularState* current, CellularState* next) {
    next->rule = current->rule;
    
    for (int i = 0; i < CA_WIDTH; i++) {
        uint8_t left = current->cells[(i - 1 + CA_WIDTH) % CA_WIDTH];
        uint8_t center = current->cells[i];
        uint8_t right = current->cells[(i + 1) % CA_WIDTH];
        
        // Wolfram rule lookup
        uint8_t neighborhood = ((left > 127) << 2) | ((center > 127) << 1) | (right > 127);
        uint8_t new_state = (current->rule >> neighborhood) & 1;
        
        next->cells[i] = new_state ? 255 : 0;
    }
}

CellularState* evolve_ca(CellularState* initial, size_t generations, size_t* out_count) {
    printf("  [ca_evolve] Evolving cellular automaton for %zu generations\n", generations);
    
    CellularState* states = malloc(generations * sizeof(CellularState));
    
    states[0] = *initial;
    printf("  [ca_evolve] Generation 0 (initial state)\n");
    
    for (size_t i = 1; i < generations; i++) {
        ca_step(&states[i-1], &states[i]);
        
        // Show progress every 10 generations
        if (i % 10 == 0 || i < 5) {
            int live_cells = 0;
            for (int j = 0; j < CA_WIDTH; j++) {
                if (states[i].cells[j] > 127) live_cells++;
            }
            printf("  [ca_evolve] Generation %zu: %d live cells\n", i, live_cells);
        }
    }
    
    printf("  [ca_evolve] Evolution complete\n\n");
    
    *out_count = generations;
    return states;
}

// ============================================================================
// STAGE 3: CELLULAR AUTOMATON → HOLOGRAPHIC ENCODING
// ============================================================================

HolographicPattern* ca_to_hologram(CellularState* states, size_t gen_count) {
    printf("  [ca→holo] Creating holographic pattern from %zu CA generations\n", gen_count);
    
    HolographicPattern* holo = malloc(sizeof(HolographicPattern));
    memset(holo, 0, sizeof(HolographicPattern));
    
    int total_pixels = 0;
    double total_intensity = 0.0;
    
    // Encode CA evolution into holographic pattern
    for (int y = 0; y < HOLO_SIZE && y < (int)gen_count; y++) {
        for (int x = 0; x < HOLO_SIZE && x < CA_WIDTH; x++) {
            double cell_val = states[y].cells[x] / 255.0;
            
            // Create interference pattern
            double phase_val = 2.0 * M_PI * cell_val;
            holo->intensity[y][x] = cell_val * cell_val;
            holo->phase[y][x] = phase_val;
            holo->amplitude[y][x] = cell_val;
            
            total_intensity += holo->intensity[y][x];
            total_pixels++;
        }
        
        if (y % 20 == 0) {
            printf("    Row %d encoded\n", y);
        }
    }
    
    double avg_intensity = total_intensity / total_pixels;
    printf("  [ca→holo] Average intensity: %.6f\n", avg_intensity);
    printf("  [ca→holo] Total pixels encoded: %d\n\n", total_pixels);
    
    return holo;
}

// ============================================================================
// STAGE 4: HOLOGRAPHIC → ASCII RENDERING
// ============================================================================

const char ASCII_RAMP[] = " .:-=+*#%@";
#define ASCII_RAMP_LEN 10

char* hologram_to_ascii(HolographicPattern* holo, size_t* out_size) {
    size_t width = HOLO_SIZE;
    size_t height = HOLO_SIZE / 2;  // Compress vertically
    
    printf("  [holo→ascii] Rendering %zux%zu hologram to ASCII\n", width, height);
    printf("  [holo→ascii] Character ramp: '%s'\n", ASCII_RAMP);
    
    size_t buffer_size = (width + 1) * height + 1;
    char* output = malloc(buffer_size);
    size_t pos = 0;
    
    int char_histogram[ASCII_RAMP_LEN] = {0};
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            double intensity = holo->intensity[y * 2][x];
            int char_idx = (int)(intensity * (ASCII_RAMP_LEN - 1));
            if (char_idx < 0) char_idx = 0;
            if (char_idx >= ASCII_RAMP_LEN) char_idx = ASCII_RAMP_LEN - 1;
            
            output[pos++] = ASCII_RAMP[char_idx];
            char_histogram[char_idx]++;
        }
        output[pos++] = '\n';
    }
    output[pos] = '\0';
    
    printf("  [holo→ascii] Character distribution:\n");
    for (int i = 0; i < ASCII_RAMP_LEN; i++) {
        printf("    '%c': %d (%.1f%%)\n", 
               ASCII_RAMP[i], 
               char_histogram[i],
               100.0 * char_histogram[i] / (width * height));
    }
    printf("\n");
    
    *out_size = pos;
    return output;
}

// ============================================================================
// MAIN PIPELINE EXECUTION
// ============================================================================

TransmutationPipeline* pipeline_create() {
    printf("[pipeline] Creating new transmutation pipeline\n\n");
    TransmutationPipeline* p = malloc(sizeof(TransmutationPipeline));
    memset(p, 0, sizeof(TransmutationPipeline));
    return p;
}

void pipeline_destroy(TransmutationPipeline* p) {
    if (!p) return;
    printf("[pipeline] Destroying pipeline and freeing memory\n");
    free(p->source_data);
    free(p->quaternion_seq);
    free(p->ca_states);
    free(p->holo_pattern);
    free(p->ascii_output);
    free(p);
}

int pipeline_execute(TransmutationPipeline* p, const char* source_data, size_t source_size) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║           TRANSMUTATION PIPELINE - VERBOSE MODE                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    // Stage 1: Parse source → Quaternion space
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ STAGE 1: Musical Notation → Quaternion Gesture Space           │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    p->quaternion_seq = parse_musical_data(source_data, source_size, &p->quat_count);
    printf("✓ Generated %zu quaternion gestures\n\n", p->quat_count);
    
    // Stage 2: Quaternion → Cellular Automaton
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ STAGE 2: Quaternion Space → Cellular Automaton                 │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    CellularState initial_ca;
    quaternion_to_ca_seed(p->quaternion_seq, p->quat_count, &initial_ca);
    
    p->ca_states = evolve_ca(&initial_ca, CA_GENERATIONS, &p->ca_gen_count);
    printf("✓ Evolved %zu generations using Wolfram rule %d\n\n", 
           p->ca_gen_count, initial_ca.rule);
    
    // Stage 3: CA → Holographic encoding
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ STAGE 3: Cellular Automaton → Holographic Pattern              │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    p->holo_pattern = ca_to_hologram(p->ca_states, p->ca_gen_count);
    printf("✓ Generated %dx%d holographic interference pattern\n\n", HOLO_SIZE, HOLO_SIZE);
    
    // Stage 4: Hologram → ASCII rendering
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ STAGE 4: Holographic Pattern → ASCII Visualization             │\n");
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    p->ascii_output = hologram_to_ascii(p->holo_pattern, &p->ascii_size);
    printf("✓ Created %zu character ASCII output\n\n", p->ascii_size);
    
    return 0;
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

int main(int argc, char** argv) {
    // Example musical data (freq,duration,amplitude)
    // These represent a simple ascending scale
    const char* musical_input = 
        "261.63,0.5,0.8\n"   // C4
        "293.66,0.5,0.7\n"   // D4
        "329.63,0.5,0.9\n"   // E4
        "349.23,0.5,0.6\n"   // F4
        "392.00,0.5,0.8\n"   // G4
        "440.00,0.5,1.0\n"   // A4
        "493.88,0.5,0.7\n"   // B4
        "523.25,1.0,0.9\n";  // C5
    
    TransmutationPipeline* pipeline = pipeline_create();
    
    pipeline_execute(pipeline, musical_input, strlen(musical_input));
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                  TRANSMUTATION COMPLETE                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("ASCII Output:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("%s", pipeline->ascii_output);
    printf("─────────────────────────────────────────────────────────────────\n\n");
    
    // Show detailed intermediate data
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    DEBUG INFORMATION                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("First Quaternion:\n");
    printf("  w = %.6f\n", pipeline->quaternion_seq[0].rotation.w);
    printf("  x = %.6f\n", pipeline->quaternion_seq[0].rotation.x);
    printf("  y = %.6f\n", pipeline->quaternion_seq[0].rotation.y);
    printf("  z = %.6f\n", pipeline->quaternion_seq[0].rotation.z);
    printf("  timestamp = %.3fs\n", pipeline->quaternion_seq[0].timestamp);
    printf("  magnitude = %.3f\n\n", pipeline->quaternion_seq[0].magnitude);
    
    printf("Cellular Automaton:\n");
    printf("  Rule: %d\n", pipeline->ca_states[0].rule);
    printf("  First generation pattern (first 64 cells):\n  ");
    for (int i = 0; i < 64; i++) {
        printf("%c", pipeline->ca_states[0].cells[i] > 127 ? '#' : '.');
        if (i == 31) printf("\n  ");
    }
    printf("\n\n");
    
    printf("Memory Usage:\n");
    printf("  Quaternion sequence: %zu bytes\n", 
           pipeline->quat_count * sizeof(GesturePoint));
    printf("  CA states: %zu bytes\n", 
           pipeline->ca_gen_count * sizeof(CellularState));
    printf("  Holographic pattern: %zu bytes\n", 
           sizeof(HolographicPattern));
    printf("  ASCII output: %zu bytes\n\n", 
           pipeline->ascii_size);
    
    pipeline_destroy(pipeline);
    
    printf("✓ Pipeline destroyed, all memory freed\n");
    
    return 0;
}
