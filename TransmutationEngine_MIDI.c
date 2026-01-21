#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * TRANSMUTATION ENGINE - MIDI Version
 * 
 * A pipeline for transforming MIDI sequences between representational domains:
 * MIDI Notes (0-127) → Quaternion Space → Cellular Automaton → Holographic → ASCII
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
    uint8_t midi_note;     // Store original MIDI note
    uint8_t velocity;      // Store original velocity
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
// MIDI UTILITIES
// ============================================================================

// Convert MIDI note (0-127) to frequency in Hz
// Formula: f = 440 * 2^((n - 69) / 12)
// MIDI 69 = A4 = 440 Hz
double midi_to_frequency(uint8_t midi_note) {
    return 440.0 * pow(2.0, (midi_note - 69) / 12.0);
}

// Get note name from MIDI number
const char* midi_to_note_name(uint8_t midi_note) {
    static const char* note_names[] = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    
    static char buffer[8];
    int octave = (midi_note / 12) - 1;
    int note_index = midi_note % 12;
    
    sprintf(buffer, "%s%d", note_names[note_index], octave);
    return buffer;
}

// ============================================================================
// STAGE 1: PARSE MIDI → QUATERNION SPACE
// ============================================================================

// MIDI note to quaternion (note number → rotation in 3D space)
Quaternion midi_to_quaternion(uint8_t midi_note, double duration, double velocity_normalized) {
    // Convert MIDI to frequency for processing
    double frequency = midi_to_frequency(midi_note);
    
    // Map frequency to rotation axis
    double normalized_freq = frequency / 440.0;
    double angle = 2.0 * M_PI * normalized_freq * duration;
    
    // Use velocity to modulate axis
    double axis_x = sin(normalized_freq * M_PI);
    double axis_y = cos(normalized_freq * M_PI);
    double axis_z = velocity_normalized;
    
    // Normalize axis
    double axis_norm = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
    if (axis_norm > 1e-10) {
        axis_x /= axis_norm;
        axis_y /= axis_norm;
        axis_z /= axis_norm;
    }
    
    // Create quaternion from axis-angle
    double half_angle = angle / 2.0;
    double sin_half = sin(half_angle);
    
    return (Quaternion){
        cos(half_angle),
        axis_x * sin_half,
        axis_y * sin_half,
        axis_z * sin_half
    };
}

// Parse MIDI data: format is "midi_note,duration,velocity\n"
// midi_note: 0-127
// duration: seconds (e.g., 0.5)
// velocity: 0-127 (MIDI velocity)
GesturePoint* parse_midi_data(const char* data, size_t size, size_t* out_count) {
    size_t capacity = 1024;
    GesturePoint* points = malloc(capacity * sizeof(GesturePoint));
    size_t count = 0;
    
    const char* ptr = data;
    while (ptr < data + size) {
        int midi_note, velocity;
        double dur;
        
        if (sscanf(ptr, "%d,%lf,%d", &midi_note, &dur, &velocity) == 3) {
            // Validate MIDI ranges
            if (midi_note < 0 || midi_note > 127) {
                fprintf(stderr, "Warning: MIDI note %d out of range (0-127), skipping\n", midi_note);
                goto next_line;
            }
            
            if (velocity < 0 || velocity > 127) {
                fprintf(stderr, "Warning: Velocity %d out of range (0-127), skipping\n", velocity);
                goto next_line;
            }
            
            if (count >= capacity) {
                capacity *= 2;
                points = realloc(points, capacity * sizeof(GesturePoint));
            }
            
            double velocity_normalized = velocity / 127.0;
            
            points[count].rotation = midi_to_quaternion((uint8_t)midi_note, dur, velocity_normalized);
            points[count].timestamp = count * dur;
            points[count].magnitude = velocity_normalized;
            points[count].midi_note = (uint8_t)midi_note;
            points[count].velocity = (uint8_t)velocity;
            count++;
        }
        
        next_line:
        // Move to next line
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
    
    // Use gesture count to influence rule selection
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
    printf("=== TRANSMUTATION PIPELINE (MIDI) ===\n\n");
    
    // Stage 1: Parse MIDI → Quaternion space
    printf("[Stage 1] Parsing MIDI data into quaternion gesture space...\n");
    p->quaternion_seq = parse_midi_data(source_data, source_size, &p->quat_count);
    printf("  Generated %zu quaternion gestures\n\n", p->quat_count);
    
    // Stage 2: Quaternion → Cellular Automaton
    printf("[Stage 2] Encoding quaternions into cellular automaton...\n");
    CellularState initial_ca;
    quaternion_to_ca_seed(p->quaternion_seq, p->quat_count, &initial_ca);
    printf("  Using Wolfram rule %d\n", initial_ca.rule);
    
    p->ca_states = evolve_ca(&initial_ca, CA_GENERATIONS, &p->ca_gen_count);
    printf("  Evolved %zu generations\n\n", p->ca_gen_count);
    
    // Stage 3: CA → Holographic encoding
    printf("[Stage 3] Creating holographic interference pattern...\n");
    p->holo_pattern = ca_to_hologram(p->ca_states, p->ca_gen_count);
    printf("  Generated %dx%d holographic pattern\n\n", HOLO_SIZE, HOLO_SIZE);
    
    // Stage 4: Hologram → ASCII rendering
    printf("[Stage 4] Rendering hologram as ASCII art...\n");
    p->ascii_output = hologram_to_ascii(p->holo_pattern, &p->ascii_size);
    printf("  Created %zu character ASCII output\n\n", p->ascii_size);
    
    return 0;
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

int main(int argc, char** argv) {
    // Example MIDI data (midi_note,duration,velocity)
    // C Major scale using MIDI note numbers
    const char* midi_input = 
        "60,0.5,100\n"   // C4 (Middle C)
        "62,0.5,90\n"    // D4
        "64,0.5,110\n"   // E4
        "65,0.5,80\n"    // F4
        "67,0.5,100\n"   // G4
        "69,0.5,127\n"   // A4 (440 Hz)
        "71,0.5,90\n"    // B4
        "72,1.0,110\n";  // C5
    
    TransmutationPipeline* pipeline = pipeline_create();
    
    pipeline_execute(pipeline, midi_input, strlen(midi_input));
    
    printf("=== TRANSMUTATION COMPLETE ===\n\n");
    printf("ASCII Output:\n");
    printf("%s\n", pipeline->ascii_output);
    
    // Show MIDI-specific debug info
    printf("\n=== MIDI DEBUG INFO ===\n");
    printf("First MIDI note: %d (%s)\n", 
           pipeline->quaternion_seq[0].midi_note,
           midi_to_note_name(pipeline->quaternion_seq[0].midi_note));
    printf("  Frequency: %.2f Hz\n", 
           midi_to_frequency(pipeline->quaternion_seq[0].midi_note));
    printf("  Velocity: %d/127\n", pipeline->quaternion_seq[0].velocity);
    printf("  Quaternion: (%.3f, %.3f, %.3f, %.3f)\n",
           pipeline->quaternion_seq[0].rotation.w,
           pipeline->quaternion_seq[0].rotation.x,
           pipeline->quaternion_seq[0].rotation.y,
           pipeline->quaternion_seq[0].rotation.z);
    
    printf("\nMIDI sequence:\n");
    for (size_t i = 0; i < pipeline->quat_count; i++) {
        printf("  Note %zu: MIDI %d (%s) @ %.3fs, velocity %d\n",
               i,
               pipeline->quaternion_seq[i].midi_note,
               midi_to_note_name(pipeline->quaternion_seq[i].midi_note),
               pipeline->quaternion_seq[i].timestamp,
               pipeline->quaternion_seq[i].velocity);
    }
    
    printf("\nCA Rule: %d\n", pipeline->ca_states[0].rule);
    printf("First CA generation pattern: ");
    for (int i = 0; i < 32; i++) {
        printf("%c", pipeline->ca_states[0].cells[i] > 127 ? '#' : '.');
    }
    printf("...\n");
    
    pipeline_destroy(pipeline);
    
    return 0;
}