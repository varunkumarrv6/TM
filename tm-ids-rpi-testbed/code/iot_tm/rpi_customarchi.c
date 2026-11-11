#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <arpa/inet.h>
#include <sys/socket.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <arm_neon.h>


/* ============================================================
   NETWORK CONFIGURATION
   ============================================================ */

#define DIRECT_PORT             6000
#define LAPTOP_PORT             5001

#define NUM_PARTS               3
#define NUM_CLASSES             6

#define PACKETS_PER_PART       1000
#define TOTAL_PACKETS           (NUM_PARTS * PACKETS_PER_PART)

#define UDP_HEADER_BYTES        4
#define ENCODED_PACKET_BYTES    1393
#define UDP_PACKET_BYTES        (UDP_HEADER_BYTES + ENCODED_PACKET_BYTES)

#define DECODED_PACKET_BYTES    796

#define RULE_BYTES              66
#define RULES_PER_PACKET        12

#define HAMMING_CODEWORDS       1592


/* ============================================================
   MODEL CONFIGURATION
   ============================================================ */

#define CLAUSES_PER_CLASS       6000
#define NUM_INPUTS              50
#define INPUT_FEATURES          263
#define NUM_LITERALS            526


/* ============================================================
   PIPELINE BUFFERING
   ============================================================ */

/*
 * Number of packet buffers circulating through pipeline.
 *
 * 16 is enough to keep all four stages busy without consuming
 * excessive memory.
 */

#define PIPELINE_BUFFERS        16


/* ============================================================
   FILES
   ============================================================ */

#define BASE_DIR \
    "/home/<pi_user>/inferencing_full/solving_ecc"

#define INPUT_FILE \
    BASE_DIR "/X_50.bin"

#define ACTUAL_FILE \
    BASE_DIR "/actual.bin"

#define TMU_FILE \
    BASE_DIR "/tmu_pred.bin"


/* ============================================================
   TIMING
   ============================================================ */

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts
    );

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1e9;
}


/* ============================================================
   CPU AFFINITY
   ============================================================ */

static void pin_current_thread(int core)
{
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    int ret =
        pthread_setaffinity_np(
            pthread_self(),
            sizeof(cpu_set_t),
            &cpuset
        );

    if (ret != 0)
    {
        fprintf(
            stderr,
            "[WARN] Could not pin thread to core %d\n",
            core
        );
    }
}


/* ============================================================
   PACKET SLOT
   ============================================================ */

typedef struct
{
    /*
     * Original UDP packet payload.
     *
     * Header is stored separately.
     */

    uint8_t encoded[
        ENCODED_PACKET_BYTES
    ];


    /*
     * Four-byte application header.
     */

    uint8_t header[
        UDP_HEADER_BYTES
    ];


    /*
     * Hamming stage output.
     *
     * One 7-bit Hamming codeword stored in
     * one byte.
     */

    uint8_t codewords[
        HAMMING_CODEWORDS
    ];


    /*
     * Syndrome for every codeword.
     *
     * 0 = clean
     * 1..7 = detected error position
     */

    uint8_t syndromes[
        HAMMING_CODEWORDS
    ];


    /*
     * Corrected 796-byte packet.
     *
     * 4-byte header is NOT included.
     */

    uint8_t decoded[
        DECODED_PACKET_BYTES
    ];


    /*
     * Packet metadata.
     */

    uint16_t sequence;

    uint8_t class_id;

    int8_t weight;

    uint8_t part_id;


    /*
     * Timing.
     */

    double decode_start;
    double decode_end;

    double correction_start;
    double correction_end;

    double inference_start;
    double inference_end;


    /*
     * Error statistics.
     */

    uint32_t detected_errors;

    uint32_t corrected_errors;

} PacketSlot;


/* ============================================================
   QUEUE
   ============================================================ */

typedef struct
{
    int items[PIPELINE_BUFFERS];

    int head;
    int tail;
    int count;

    int closed;

    pthread_mutex_t mutex;

    pthread_cond_t not_empty;

    pthread_cond_t not_full;

} Queue;


/* ============================================================
   QUEUE INITIALIZATION
   ============================================================ */

static void queue_init(Queue *q)
{
    memset(
        q,
        0,
        sizeof(Queue)
    );

    pthread_mutex_init(
        &q->mutex,
        NULL
    );

    pthread_cond_init(
        &q->not_empty,
        NULL
    );

    pthread_cond_init(
        &q->not_full,
        NULL
    );
}


/* ============================================================
   QUEUE CLOSE
   ============================================================ */

static void queue_close(Queue *q)
{
    pthread_mutex_lock(
        &q->mutex
    );

    q->closed = 1;

    pthread_cond_broadcast(
        &q->not_empty
    );

    pthread_cond_broadcast(
        &q->not_full
    );

    pthread_mutex_unlock(
        &q->mutex
    );
}


/* ============================================================
   QUEUE PUSH
   ============================================================ */

static int queue_push(
    Queue *q,
    int value
)
{
    pthread_mutex_lock(
        &q->mutex
    );

    while (
        q->count == PIPELINE_BUFFERS &&
        !q->closed
    )
    {
        pthread_cond_wait(
            &q->not_full,
            &q->mutex
        );
    }

    if (q->closed)
    {
        pthread_mutex_unlock(
            &q->mutex
        );

        return -1;
    }

    q->items[q->tail] =
        value;

    q->tail =
        (q->tail + 1) %
        PIPELINE_BUFFERS;

    q->count++;

    pthread_cond_signal(
        &q->not_empty
    );

    pthread_mutex_unlock(
        &q->mutex
    );

    return 0;
}


/* ============================================================
   QUEUE POP
   ============================================================ */

static int queue_pop(
    Queue *q,
    int *value
)
{
    pthread_mutex_lock(
        &q->mutex
    );

    while (
        q->count == 0 &&
        !q->closed
    )
    {
        pthread_cond_wait(
            &q->not_empty,
            &q->mutex
        );
    }

    if (
        q->count == 0 &&
        q->closed
    )
    {
        pthread_mutex_unlock(
            &q->mutex
        );

        return -1;
    }

    *value =
        q->items[q->head];

    q->head =
        (q->head + 1) %
        PIPELINE_BUFFERS;

    q->count--;

    pthread_cond_signal(
        &q->not_full
    );

    pthread_mutex_unlock(
        &q->mutex
    );

    return 0;
}


/* ============================================================
   GLOBAL PIPELINE
   ============================================================ */

static PacketSlot slots[
    PIPELINE_BUFFERS
];


/*
 * Free buffers
 */

static Queue free_queue;


/*
 * RECEIVE → DECODE
 */

static Queue decode_queue;


/*
 * DECODE → CORRECTION
 */

static Queue correction_queue;


/*
 * CORRECTION → INFERENCE
 */

static Queue inference_queue;


/* ============================================================
   GLOBAL TIMING
   ============================================================ */

static double pipeline_start;
static double pipeline_end;

static double rx_first;
static double rx_last;

static double decode_first = 0.0;
static double decode_last = 0.0;

static double correction_first = 0.0;
static double correction_last = 0.0;

static double inference_first = 0.0;
static double inference_last = 0.0;


/*
 * Cumulative CPU time spent inside each stage.
 */

static double decode_cpu_time = 0.0;
static double correction_cpu_time = 0.0;
static double inference_cpu_time = 0.0;


/* ============================================================
   STATISTICS
   ============================================================ */

static uint64_t total_received = 0;

static uint64_t total_detected_errors = 0;

static uint64_t total_corrected_errors = 0;


/* ============================================================
   DUPLICATE DETECTION
   ============================================================ */

static uint8_t packet_seen[
    NUM_PARTS
][
    PACKETS_PER_PART
];


/* ============================================================
   INPUT DATA
   ============================================================ */

/*
 * Original X_50.bin:
 *
 * 50 samples × 33 bytes
 */

static uint8_t input_raw[
    NUM_INPUTS
][
    33
];


/*
 * Packed 526-literal representation.
 *
 * 66 bytes/sample.
 */

static uint8_t input_packed[
    NUM_INPUTS
][
    RULE_BYTES
];


/*
 * Transposed input for SIMD:
 *
 * input_vector[byte][sample_group]
 *
 * 50 samples padded to 64.
 */

static uint8x16_t input_vectors[
    RULE_BYTES
][
    4
];


/* ============================================================
   SCORES
   ============================================================ */

/*
 * Maximum score is ±6000.
 *
 * int16_t is sufficient.
 */

static int16_t scores[
    NUM_INPUTS
][
    NUM_CLASSES
];


/* ============================================================
   GROUND TRUTH / REFERENCE
   ============================================================ */

static uint8_t actual[
    NUM_INPUTS
];

static uint8_t tmu_prediction[
    NUM_INPUTS
];


/* ============================================================
   LOAD INPUT FILES
   ============================================================ */

static int load_inputs(void)
{
    FILE *f;


    /* ---------------- X_50.bin ---------------- */

    f = fopen(
        INPUT_FILE,
        "rb"
    );

    if (!f)
    {
        perror(
            "X_50.bin"
        );

        return -1;
    }

    size_t n =
        fread(
            input_raw,
            1,
            NUM_INPUTS * 33,
            f
        );

    fclose(f);

    if (
        n !=
        NUM_INPUTS * 33
    )
    {
        fprintf(
            stderr,
            "ERROR: X_50.bin size\n"
        );

        return -1;
    }


    /* ---------------- ACTUAL ---------------- */

    f = fopen(
        ACTUAL_FILE,
        "rb"
    );

    if (!f)
    {
        perror(
            "actual.bin"
        );

        return -1;
    }

    n =
        fread(
            actual,
            1,
            NUM_INPUTS,
            f
        );

    fclose(f);

    if (n != NUM_INPUTS)
    {
        fprintf(
            stderr,
            "ERROR: actual.bin\n"
        );

        return -1;
    }


    /* ---------------- TMU ---------------- */

    f = fopen(
        TMU_FILE,
        "rb"
    );

    if (!f)
    {
        perror(
            "tmu_pred.bin"
        );

        return -1;
    }

    n =
        fread(
            tmu_prediction,
            1,
            NUM_INPUTS,
            f
        );

    fclose(f);

    if (n != NUM_INPUTS)
    {
        fprintf(
            stderr,
            "ERROR: tmu_pred.bin\n"
        );

        return -1;
    }


    return 0;
}


/* ============================================================
   BUILD 526 LITERALS
   ============================================================ */

static void build_input_packed(void)
{
    memset(
        input_packed,
        0,
        sizeof(input_packed)
    );


    for (int sample = 0;
         sample < NUM_INPUTS;
         sample++)
    {
        for (int feature = 0;
             feature < INPUT_FEATURES;
             feature++)
        {
            int byte_index =
                feature / 8;

            int bit_index =
                feature % 8;

            uint8_t bit =
                (
                    input_raw[sample][byte_index]
                    >>
                    bit_index
                ) & 1;


            /*
             * Positive literal.
             */

            if (bit)
            {
                int literal =
                    feature;

                input_packed[sample]
                    [literal / 8]
                    |=
                    (uint8_t)(
                        1u <<
                        (literal % 8)
                    );
            }


            /*
             * Negative literal.
             */

            if (!bit)
            {
                int literal =
                    INPUT_FEATURES +
                    feature;

                input_packed[sample]
                    [literal / 8]
                    |=
                    (uint8_t)(
                        1u <<
                        (literal % 8)
                    );
            }
        }


        /*
         * 526 bits occupy 66 bytes.
         *
         * Last byte:
         *
         * 6 valid bits
         * 2 padding bits
         */

        input_packed[sample][65]
            &= 0x3F;
    }
}


/* ============================================================
   TRANSPOSE INPUT FOR NEON
   ============================================================ */

static void prepare_neon_inputs(void)
{
    uint8_t temp[
        RULE_BYTES
    ][
        64
    ];

    memset(
        temp,
        0,
        sizeof(temp)
    );


    for (int sample = 0;
         sample < NUM_INPUTS;
         sample++)
    {
        for (int b = 0;
             b < RULE_BYTES;
             b++)
        {
            temp[b][sample] =
                input_packed[
                    sample
                ][b];
        }
    }


    /*
     * Four 128-bit vectors:
     *
     * samples 0-15
     * samples 16-31
     * samples 32-47
     * samples 48-63
     */

    for (int b = 0;
         b < RULE_BYTES;
         b++)
    {
        input_vectors[b][0] =
            vld1q_u8(
                &temp[b][0]
            );

        input_vectors[b][1] =
            vld1q_u8(
                &temp[b][16]
            );

        input_vectors[b][2] =
            vld1q_u8(
                &temp[b][32]
            );

        input_vectors[b][3] =
            vld1q_u8(
                &temp[b][48]
            );
    }
}


/* ============================================================
   FAST HAMMING CODEWORD EXTRACTION
   ============================================================ */

/*
 * Encoded stream:
 *
 * 7-bit codewords packed continuously.
 *
 * bitorder = BIG
 */

static inline uint8_t extract_codeword(
    const uint8_t *data,
    size_t codeword_index
)
{
    size_t bit_position =
        codeword_index * 7;

    size_t byte_index =
        bit_position >> 3;

    int offset =
        bit_position & 7;


    uint16_t word =
        (uint16_t)data[byte_index]
        << 8;


    if (
        byte_index + 1 <
        ENCODED_PACKET_BYTES
    )
    {
        word |=
            data[byte_index + 1];
    }


    /*
     * Extract exactly 7 bits.
     */

    return (uint8_t)(
        (word >>
         (9 - offset))
        & 0x7F
    );
}


/* ============================================================
   HAMMING SYNDROME
   ============================================================ */

/*
 * Codeword layout:
 *
 * bit 6 5 4 3 2 1 0
 *
 * P1 P2 D1 P4 D2 D3 D4
 */

static inline uint8_t calculate_syndrome(
    uint8_t cw
)
{
    int p1 =
        (cw >> 6) & 1;

    int p2 =
        (cw >> 5) & 1;

    int d1 =
        (cw >> 4) & 1;

    int p4 =
        (cw >> 3) & 1;

    int d2 =
        (cw >> 2) & 1;

    int d3 =
        (cw >> 1) & 1;

    int d4 =
        cw & 1;


    int s1 =
        p1 ^
        d1 ^
        d2 ^
        d4;

    int s2 =
        p2 ^
        d1 ^
        d3 ^
        d4;

    int s4 =
        p4 ^
        d2 ^
        d3 ^
        d4;


    return (uint8_t)(
        s1 |
        (s2 << 1) |
        (s4 << 2)
    );
}


/* ============================================================
   STAGE 1
   HAMMING DECODE + ERROR DETECTION
   ============================================================ */

/*
 * IMPORTANT:
 *
 * This stage DOES NOT correct the codeword.
 *
 * It only:
 *
 * 1. extracts 7-bit codeword
 * 2. calculates syndrome
 * 3. stores syndrome
 *
 * This allows us to measure decoding/error-detection
 * independently from correction.
 */

static void decode_packet(
    PacketSlot *slot
)
{
    for (size_t i = 0;
         i < HAMMING_CODEWORDS;
         i++)
    {
        uint8_t cw =
            extract_codeword(
                slot->encoded,
                i
            );


        uint8_t syndrome =
            calculate_syndrome(
                cw
            );


        slot->codewords[i] =
            cw;

        slot->syndromes[i] =
            syndrome;


        if (syndrome != 0)
        {
            slot->detected_errors++;
        }
    }
}


/* ============================================================
   STAGE 2
   ERROR CORRECTION + BYTE RECONSTRUCTION
   ============================================================ */

static void correct_packet(
    PacketSlot *slot
)
{
    slot->corrected_errors = 0;


    /*
     * Two Hamming codewords produce
     * one original byte.
     */

    for (size_t byte_index = 0;
         byte_index < DECODED_PACKET_BYTES;
         byte_index++)
    {
        size_t cw_index =
            byte_index * 2;


        uint8_t high =
            slot->codewords[
                cw_index
            ];

        uint8_t low =
            slot->codewords[
                cw_index + 1
            ];


        uint8_t syndrome_high =
            slot->syndromes[
                cw_index
            ];

        uint8_t syndrome_low =
            slot->syndromes[
                cw_index + 1
            ];


        /*
         * Correct HIGH codeword.
         */

        if (syndrome_high != 0)
        {
            int bit_to_flip =
                7 -
                syndrome_high;


            if (
                bit_to_flip >= 0 &&
                bit_to_flip <= 6
            )
            {
                high ^=
                    (uint8_t)(
                        1u <<
                        bit_to_flip
                    );

                slot->corrected_errors++;
            }
        }


        /*
         * Correct LOW codeword.
         */

        if (syndrome_low != 0)
        {
            int bit_to_flip =
                7 -
                syndrome_low;


            if (
                bit_to_flip >= 0 &&
                bit_to_flip <= 6
            )
            {
                low ^=
                    (uint8_t)(
                        1u <<
                        bit_to_flip
                    );

                slot->corrected_errors++;
            }
        }


        /*
         * Extract D1 D2 D3 D4.
         */

        uint8_t high_data =
            (
                ((high >> 4) & 1) << 3
            ) |
            (
                ((high >> 2) & 1) << 2
            ) |
            (
                ((high >> 1) & 1) << 1
            ) |
            (
                high & 1
            );


        uint8_t low_data =
            (
                ((low >> 4) & 1) << 3
            ) |
            (
                ((low >> 2) & 1) << 2
            ) |
            (
                ((low >> 1) & 1) << 1
            ) |
            (
                low & 1
            );


        slot->decoded[
            byte_index
        ] =
            (uint8_t)(
                (high_data << 4) |
                low_data
            );
    }
}


/* ============================================================
   NEON CLAUSE EVALUATION
   ============================================================ */

/*
 * Evaluate one rule against 50 inputs.
 *
 * Four 128-bit vectors represent:
 *
 *   lanes 0-15
 *   lanes 16-31
 *   lanes 32-47
 *   lanes 48-63
 *
 * Only lanes 0-49 are real samples.
 *
 * Result:
 *
 * match[64]
 *
 * 1 = clause matches
 * 0 = clause does not match
 */

static inline void clause_match_neon(
    const uint8_t *rule,
    uint8_t match[64]
)
{
    uint8x16_t bad0 =
        vdupq_n_u8(0);

    uint8x16_t bad1 =
        vdupq_n_u8(0);

    uint8x16_t bad2 =
        vdupq_n_u8(0);

    uint8x16_t bad3 =
        vdupq_n_u8(0);


    /*
     * Check whether clause is empty.
     */

    uint8_t nonempty = 0;

    for (int b = 0;
         b < RULE_BYTES;
         b++)
    {
        nonempty |=
            rule[b];
    }


    /*
     * Empty clauses are disabled in the
     * Python reference implementation.
     */

    if (nonempty == 0)
    {
        memset(
            match,
            0,
            64
        );

        return;
    }


    /*
     * Exact Python operation:
     *
     * required_bits =
     *      rule & ~input
     *
     * Clause matches when all required bits
     * are zero.
     */

    for (int b = 0;
         b < RULE_BYTES;
         b++)
    {
        uint8x16_t rule_vec =
            vdupq_n_u8(
                rule[b]
            );


        uint8x16_t input0 =
            input_vectors[b][0];

        uint8x16_t input1 =
            input_vectors[b][1];

        uint8x16_t input2 =
            input_vectors[b][2];

        uint8x16_t input3 =
            input_vectors[b][3];


        uint8x16_t req0 =
            vandq_u8(
                rule_vec,
                vmvnq_u8(input0)
            );

        uint8x16_t req1 =
            vandq_u8(
                rule_vec,
                vmvnq_u8(input1)
            );

        uint8x16_t req2 =
            vandq_u8(
                rule_vec,
                vmvnq_u8(input2)
            );

        uint8x16_t req3 =
            vandq_u8(
                rule_vec,
                vmvnq_u8(input3)
            );


        bad0 =
            vorrq_u8(
                bad0,
                req0
            );

        bad1 =
            vorrq_u8(
                bad1,
                req1
            );

        bad2 =
            vorrq_u8(
                bad2,
                req2
            );

        bad3 =
            vorrq_u8(
                bad3,
                req3
            );
    }


    uint8x16_t zero =
        vdupq_n_u8(0);


    uint8x16_t m0 =
        vceqq_u8(
            bad0,
            zero
        );

    uint8x16_t m1 =
        vceqq_u8(
            bad1,
            zero
        );

    uint8x16_t m2 =
        vceqq_u8(
            bad2,
            zero
        );

    uint8x16_t m3 =
        vceqq_u8(
            bad3,
            zero
        );


    vst1q_u8(
        &match[0],
        m0
    );

    vst1q_u8(
        &match[16],
        m1
    );

    vst1q_u8(
        &match[32],
        m2
    );

    vst1q_u8(
        &match[48],
        m3
    );
}


/* ============================================================
   STAGE 3
   NEON TM INFERENCE
   ============================================================ */

static void infer_packet(
    PacketSlot *slot
)
{
    uint8_t matches[64];


    int class_id =
        slot->class_id;

    int weight =
        slot->weight;


    /*
     * 12 complete rules.
     */

    for (int r = 0;
         r < RULES_PER_PACKET;
         r++)
    {
        const uint8_t *rule =
            slot->decoded +
            r * RULE_BYTES;


        clause_match_neon(
            rule,
            matches
        );


        /*
         * Convert the 0xFF/0x00 NEON result
         * into +1/-1 score contribution.
         *
         * Only samples 0..49 are used.
         */

        for (int sample = 0;
             sample < NUM_INPUTS;
             sample++)
        {
            if (matches[sample])
            {
                scores[sample][class_id] +=
                    (int16_t)weight;
            }
        }
    }
}


/* ============================================================
   DECODER THREAD
   CORE 1
   ============================================================ */

static void *decoder_thread(
    void *arg
)
{
    (void)arg;

    pin_current_thread(1);


    int slot_id;


    while (
        queue_pop(
            &decode_queue,
            &slot_id
        ) == 0
    )
    {
        PacketSlot *slot =
            &slots[slot_id];


        slot->decode_start =
            now_seconds();


        if (decode_first == 0.0)
            decode_first =
                slot->decode_start;


        slot->detected_errors = 0;


        decode_packet(
            slot
        );


        slot->decode_end =
            now_seconds();


        decode_last =
            slot->decode_end;


        decode_cpu_time +=
            slot->decode_end -
            slot->decode_start;


        total_detected_errors +=
            slot->detected_errors;


        queue_push(
            &correction_queue,
            slot_id
        );
    }


    /*
     * Tell correction stage that no more
     * packets will arrive.
     */

    queue_close(
        &correction_queue
    );


    return NULL;
}


/* ============================================================
   CORRECTION THREAD
   CORE 2
   ============================================================ */

static void *correction_thread(
    void *arg
)
{
    (void)arg;

    pin_current_thread(2);


    int slot_id;


    while (
        queue_pop(
            &correction_queue,
            &slot_id
        ) == 0
    )
    {
        PacketSlot *slot =
            &slots[slot_id];


        slot->correction_start =
            now_seconds();


        if (correction_first == 0.0)
            correction_first =
                slot->correction_start;


        correct_packet(
            slot
        );


        slot->correction_end =
            now_seconds();


        correction_last =
            slot->correction_end;


        correction_cpu_time +=
            slot->correction_end -
            slot->correction_start;


        total_corrected_errors +=
            slot->corrected_errors;


        queue_push(
            &inference_queue,
            slot_id
        );
    }


    queue_close(
        &inference_queue
    );


    return NULL;
}


/* ============================================================
   INFERENCE THREAD
   CORE 3
   ============================================================ */

static void *inference_thread(
    void *arg
)
{
    (void)arg;

    pin_current_thread(3);


    int slot_id;


    while (
        queue_pop(
            &inference_queue,
            &slot_id
        ) == 0
    )
    {
        PacketSlot *slot =
            &slots[slot_id];


        slot->inference_start =
            now_seconds();


        if (inference_first == 0.0)
            inference_first =
                slot->inference_start;


        infer_packet(
            slot
        );


        slot->inference_end =
            now_seconds();


        inference_last =
            slot->inference_end;


        inference_cpu_time +=
            slot->inference_end -
            slot->inference_start;


        /*
         * Packet buffer is now free.
         */

        queue_push(
            &free_queue,
            slot_id
        );
    }


    return NULL;
}


/* ============================================================
   UDP SOCKET
   ============================================================ */

static int create_udp_socket(
    int port
)
{
    int fd =
        socket(
            AF_INET,
            SOCK_DGRAM,
            0
        );


    if (fd < 0)
    {
        perror(
            "socket"
        );

        exit(EXIT_FAILURE);
    }


    int yes = 1;

    setsockopt(
        fd,
        SOL_SOCKET,
        SO_REUSEADDR,
        &yes,
        sizeof(yes)
    );


    /*
     * Large receive buffer.
     */

    int receive_buffer =
        8 * 1024 * 1024;

    setsockopt(
        fd,
        SOL_SOCKET,
        SO_RCVBUF,
        &receive_buffer,
        sizeof(receive_buffer)
    );


    struct sockaddr_in addr;

    memset(
        &addr,
        0,
        sizeof(addr)
    );


    addr.sin_family =
        AF_INET;

    addr.sin_addr.s_addr =
        htonl(INADDR_ANY);

    addr.sin_port =
        htons(port);


    if (
        bind(
            fd,
            (struct sockaddr *)&addr,
            sizeof(addr)
        ) < 0
    )
    {
        perror(
            "bind"
        );

        close(fd);

        exit(EXIT_FAILURE);
    }


    return fd;
}


/* ============================================================
   PARSE APPLICATION HEADER
   ============================================================ */

static int parse_header(
    PacketSlot *slot
)
{
    uint8_t b0 =
        slot->header[0];


    uint8_t weight_field =
        (b0 >> 4) & 0x0F;

    uint8_t class_id =
        b0 & 0x0F;


    /*
     * Current packet format only defines
     * weight field 0 or 1.
     */

    if (
        weight_field != 0 &&
        weight_field != 1
    )
    {
        return -1;
    }


    if (
        class_id >= NUM_CLASSES
    )
    {
        return -1;
    }


    uint16_t seq =
        (uint16_t)slot->header[1] |
        (
            (uint16_t)slot->header[2]
            << 8
        );


    if (
        seq >= PACKETS_PER_PART
    )
    {
        return -1;
    }


    slot->class_id =
        class_id;

    slot->sequence =
        seq;

    slot->weight =
        weight_field
            ? +1
            : -1;


    /*
     * Classes:
     *
     * 0,1 → part 0
     * 2,3 → part 1
     * 4,5 → part 2
     */

    slot->part_id =
        class_id / 2;


    return 0;
}


/* ============================================================
   FINAL SCORE REDUCTION
   ============================================================ */

static void final_inference_report(void)
{
    printf("\n");
    printf(
        "============================================================\n"
    );

    printf(
        "                    FINAL INFERENCE\n"
    );

    printf(
        "============================================================\n"
    );


    int correct_actual = 0;
    int correct_tmu = 0;


    for (int sample = 0;
         sample < NUM_INPUTS;
         sample++)
    {
        int prediction = 0;


        for (int c = 1;
             c < NUM_CLASSES;
             c++)
        {
            if (
                scores[sample][c]
                >
                scores[sample][prediction]
            )
            {
                prediction = c;
            }
        }


        if (
            prediction ==
            actual[sample]
        )
        {
            correct_actual++;
        }


        if (
            prediction ==
            tmu_prediction[sample]
        )
        {
            correct_tmu++;
        }


        printf(
            "%2d : actual=%d  TMU=%d  C=%d  "
            "scores=[%d %d %d %d %d %d]\n",

            sample,

            actual[sample],

            tmu_prediction[sample],

            prediction,

            scores[sample][0],
            scores[sample][1],
            scores[sample][2],
            scores[sample][3],
            scores[sample][4],
            scores[sample][5]
        );
    }


    printf("\n");

    printf(
        "C vs Ground Truth : %d/%d\n",
        correct_actual,
        NUM_INPUTS
    );

    printf(
        "C Accuracy        : %.2f%%\n",
        100.0 *
        (double)correct_actual /
        NUM_INPUTS
    );


    printf(
        "C vs TMU          : %d/%d\n",
        correct_tmu,
        NUM_INPUTS
    );


    if (
        correct_tmu ==
        NUM_INPUTS
    )
    {
        printf(
            "TMU comparison    : MATCH\n"
        );
    }
    else
    {
        printf(
            "TMU comparison    : MISMATCH\n"
        );
    }
}


/* ============================================================
   MAIN
   ============================================================ */

int main(void)
{
    printf("\n");

    printf(
        "============================================================\n"
    );

    printf(
        "       4-CORE PIPELINED HAMMING + NEON TM RECEIVER\n"
    );

    printf(
        "============================================================\n"
    );

    printf(
        "UDP ports              : %d / %d\n",
        DIRECT_PORT,
        LAPTOP_PORT
    );

    printf(
        "Parts                  : %d\n",
        NUM_PARTS
    );

    printf(
        "Packets / part         : %d\n",
        PACKETS_PER_PART
    );

    printf(
        "Total packets          : %d\n",
        TOTAL_PACKETS
    );

    printf(
        "Encoded payload        : %d bytes\n",
        ENCODED_PACKET_BYTES
    );

    printf(
        "Decoded payload        : %d bytes\n",
        DECODED_PACKET_BYTES
    );

    printf(
        "Hamming codewords      : %d / packet\n",
        HAMMING_CODEWORDS
    );

    printf(
        "Rules / packet         : %d\n",
        RULES_PER_PACKET
    );

    printf(
        "NEON                   : 128-bit ARM NEON\n"
    );

    printf(
        "Pipeline cores         : RX=0 DEC=1 CORR=2 INF=3\n"
    );

    printf(
        "============================================================\n"
    );


    /* --------------------------------------------------------
       LOAD INPUT
       -------------------------------------------------------- */

    if (
        load_inputs() != 0
    )
    {
        return EXIT_FAILURE;
    }


    build_input_packed();

    prepare_neon_inputs();


    memset(
        scores,
        0,
        sizeof(scores)
    );

    memset(
        packet_seen,
        0,
        sizeof(packet_seen)
    );


    /* --------------------------------------------------------
       INITIALIZE QUEUES
       -------------------------------------------------------- */

    queue_init(
        &free_queue
    );

    queue_init(
        &decode_queue
    );

    queue_init(
        &correction_queue
    );

    queue_init(
        &inference_queue
    );


    /*
     * Initially all packet slots are free.
     */

    for (int i = 0;
         i < PIPELINE_BUFFERS;
         i++)
    {
        queue_push(
            &free_queue,
            i
        );
    }


    /* --------------------------------------------------------
       CREATE WORKER THREADS
       -------------------------------------------------------- */

    pthread_t decoder;
    pthread_t correction;
    pthread_t inference;


    if (
        pthread_create(
            &decoder,
            NULL,
            decoder_thread,
            NULL
        ) != 0
    )
    {
        perror(
            "pthread_create decoder"
        );

        return EXIT_FAILURE;
    }


    if (
        pthread_create(
            &correction,
            NULL,
            correction_thread,
            NULL
        ) != 0
    )
    {
        perror(
            "pthread_create correction"
        );

        return EXIT_FAILURE;
    }


    if (
        pthread_create(
            &inference,
            NULL,
            inference_thread,
            NULL
        ) != 0
    )
    {
        perror(
            "pthread_create inference"
        );

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
       PIN MAIN THREAD TO CORE 0
       -------------------------------------------------------- */

    pin_current_thread(0);


    /* --------------------------------------------------------
       UDP SOCKETS
       -------------------------------------------------------- */

    int sock_direct =
        create_udp_socket(
            DIRECT_PORT
        );

    int sock_laptop =
        create_udp_socket(
            LAPTOP_PORT
        );


    printf("\n");
    printf(
        "[PIPELINE] READY\n"
    );

    printf(
        "[CORE 0] RX\n"
    );

    printf(
        "[CORE 1] Hamming decode + syndrome\n"
    );

    printf(
        "[CORE 2] error correction\n"
    );

    printf(
        "[CORE 3] NEON inference\n"
    );

    printf(
        "Waiting for %d unique packets...\n\n",
        TOTAL_PACKETS
    );


    /* --------------------------------------------------------
       TOTAL PIPELINE START
       -------------------------------------------------------- */

    pipeline_start =
        now_seconds();


    /* --------------------------------------------------------
       RECEIVE LOOP
       -------------------------------------------------------- */

    while (
        total_received <
        TOTAL_PACKETS
    )
    {
        fd_set readfds;

        FD_ZERO(
            &readfds
        );

        FD_SET(
            sock_direct,
            &readfds
        );

        FD_SET(
            sock_laptop,
            &readfds
        );


        int maxfd =
            sock_direct >
            sock_laptop
                ? sock_direct
                : sock_laptop;


        int ret =
            select(
                maxfd + 1,
                &readfds,
                NULL,
                NULL,
                NULL
            );


        if (ret < 0)
        {
            if (
                errno ==
                EINTR
            )
            {
                continue;
            }

            perror(
                "select"
            );

            break;
        }


        int fd =
            FD_ISSET(
                sock_direct,
                &readfds
            )
            ? sock_direct
            : sock_laptop;


        /*
         * Obtain a free pipeline buffer.
         *
         * If all 16 are busy, this naturally
         * applies back-pressure to RX.
         */

        int slot_id;

        if (
            queue_pop(
                &free_queue,
                &slot_id
            ) != 0
        )
        {
            fprintf(
                stderr,
                "Free queue closed unexpectedly\n"
            );

            break;
        }


        PacketSlot *slot =
            &slots[slot_id];


        uint8_t packet[
            UDP_PACKET_BYTES
        ];


        struct sockaddr_in src;

        socklen_t src_len =
            sizeof(src);


        ssize_t received =
            recvfrom(
                fd,
                packet,
                sizeof(packet),
                0,
                (struct sockaddr *)&src,
                &src_len
            );


        if (
            received !=
            UDP_PACKET_BYTES
        )
        {
            /*
             * Invalid packet size.
             */

            queue_push(
                &free_queue,
                slot_id
            );

            continue;
        }


        /*
         * Record first RX time.
         */

        double rx_time =
            now_seconds();


        if (total_received == 0)
        {
            rx_first =
                rx_time;
        }


        rx_last =
            rx_time;


        /*
         * Copy 4-byte application header.
         */

        memcpy(
            slot->header,
            packet,
            UDP_HEADER_BYTES
        );


        /*
         * Copy Hamming payload.
         */

        memcpy(
            slot->encoded,
            packet +
            UDP_HEADER_BYTES,
            ENCODED_PACKET_BYTES
        );


        /*
         * Parse header.
         */

        if (
            parse_header(
                slot
            ) != 0
        )
        {
            queue_push(
                &free_queue,
                slot_id
            );

            continue;
        }


        /*
         * Detect duplicates.
         */

        uint8_t part =
            slot->part_id;

        uint16_t seq =
            slot->sequence;


        if (
            packet_seen[part][seq]
        )
        {
            queue_push(
                &free_queue,
                slot_id
            );

            continue;
        }


        packet_seen[part][seq] =
            1;


        total_received++;


        /*
         * Print progress occasionally.
         */

        if (
            total_received % 100 ==
            0 ||
            total_received ==
            TOTAL_PACKETS
        )
        {
            printf(
                "\rReceived: %llu / %d",
                (unsigned long long)
                total_received,
                TOTAL_PACKETS
            );

            fflush(stdout);
        }


        /*
         * IMMEDIATELY hand packet to
         * Hamming decoder.
         */

        queue_push(
            &decode_queue,
            slot_id
        );
    }


    printf("\n\n");


    /* --------------------------------------------------------
       RX FINISHED
       -------------------------------------------------------- */

    printf(
        "[RX] All %d unique packets received.\n",
        TOTAL_PACKETS
    );


    /*
     * No more packets can enter decoder.
     */

    queue_close(
        &decode_queue
    );


    close(
        sock_direct
    );

    close(
        sock_laptop
    );


    /* --------------------------------------------------------
       WAIT FOR PIPELINE
       -------------------------------------------------------- */

    printf(
        "[PIPELINE] Waiting for decode/correction/inference...\n"
    );


    pthread_join(
        decoder,
        NULL
    );

    pthread_join(
        correction,
        NULL
    );

    pthread_join(
        inference,
        NULL
    );


    /* --------------------------------------------------------
       TOTAL END
       -------------------------------------------------------- */

    pipeline_end =
        now_seconds();


    /* --------------------------------------------------------
       FINAL INFERENCE
       -------------------------------------------------------- */

    final_inference_report();


    /* ========================================================
       TIMING REPORT
       ======================================================== */

    double rx_wall =
        rx_last -
        rx_first;

    double decode_wall =
        decode_last -
        decode_first;

    double correction_wall =
        correction_last -
        correction_first;

    double inference_wall =
        inference_last -
        inference_first;

    double total_wall =
        pipeline_end -
        pipeline_start;


    printf("\n");

    printf(
        "============================================================\n"
    );

    printf(
        "                    PIPELINE TIMING\n"
    );

    printf(
        "============================================================\n"
    );


    printf(
        "\n"
        "1. RECEIVE\n"
        "------------------------------------------------------------\n"
    );

    printf(
        "RX wall time          : %.3f ms\n",
        rx_wall * 1000.0
    );


    printf(
        "\n"
        "2. HAMMING DECODE + ERROR DETECTION\n"
        "------------------------------------------------------------\n"
    );

    printf(
        "Decode wall time      : %.3f ms\n",
        decode_wall * 1000.0
    );

    printf(
        "Decode CPU time       : %.3f ms\n",
        decode_cpu_time * 1000.0
    );


    printf(
        "\n"
        "3. ERROR CORRECTION\n"
        "------------------------------------------------------------\n"
    );

    printf(
        "Correction wall time  : %.3f ms\n",
        correction_wall * 1000.0
    );

    printf(
        "Correction CPU time   : %.3f ms\n",
        correction_cpu_time * 1000.0
    );


    printf(
        "\n"
        "4. NEON INFERENCE\n"
        "------------------------------------------------------------\n"
    );

    printf(
        "Inference wall time   : %.3f ms\n",
        inference_wall * 1000.0
    );

    printf(
        "Inference CPU time    : %.3f ms\n",
        inference_cpu_time * 1000.0
    );


    printf(
        "\n"
        "5. ECC STATISTICS\n"
        "------------------------------------------------------------\n"
    );

    printf(
        "Detected codewords    : %llu\n",
        (unsigned long long)
        total_detected_errors
    );

    printf(
        "Corrected codewords   : %llu\n",
        (unsigned long long)
        total_corrected_errors
    );


    printf(
        "\n"
        "6. COMPLETE PIPELINE\n"
        "------------------------------------------------------------\n"
    );

    printf(
        "Total pipeline time   : %.3f ms\n",
        total_wall * 1000.0
    );

    printf(
        "Total pipeline time   : %.6f sec\n",
        total_wall
    );


    printf(
        "\n"
        "============================================================\n"
    );


    printf(
        "                         DONE\n"
    );

    printf(
        "============================================================\n"
    );


    return 0;
}