/* The Echo's noise reduction (src/hassmic/micdenoise.c, the very code the device runs) on a WAV file, so a strength
 * or mode can be judged offline on recordings Home Assistant kept (issue #131): the same sentence through Off and
 * through each variant, into the same speech-to-text, without saying it again.
 *
 * usage: denoise-file [-c dB] [-s] [-b] [-w s] [-W] [-v file] [-g dB] in.wav out.wav
 *   -c dB   the cap (6, 9, 12 = Low, Medium, High; 9 unless given)
 *   -s      MICDENOISE_SPEECH_SAFE, -b MICDENOISE_BAND (both: the two together)
 *   -w s    warm-up: the last s seconds of the file (room noise after the sentence, 0.5 unless given) go in first with
 *           the output thrown away, as hassmic feeds RNNoise the second before a command from the ring.  0: cold.
 *   -W      warm up on the start of the file instead (what precedes the wake word, where the recording has it)
 *   -v file per frame: RNNoise's speech probability and the cap in force, for tuning the speech-safe constants
 *   -g dB   the file taken down by that much ahead of the reduction and back up behind it: Home Assistant's
 *           recordings are behind hassmic's gain (speech about -25 dBFS), the device feeds RNNoise the stream ahead
 *           of it (-55 dBFS), so -g 30 plays the device's level
 *
 * The file: 16 kHz, mono, 16-bit PCM, as Home Assistant's debug recordings are (`debug_recording_dir`).  A run of
 * zeros at the start (hassmic's 800 ms lead-in, which goes to Home Assistant past the reduction) is kept as it is and
 * the reduction starts after it.  The output is lined up with the input (the chain's delay removed, the tail flushed),
 * so a sample of the output is the same moment as that sample of the input: metrics on the two compare directly. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "micdenoise.h"

static uint32_t le32(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const unsigned char *p) { return p[0] | p[1] << 8; }
static void put32(FILE *f, uint32_t v) { unsigned char p[4] = { v, v >> 8, v >> 16, v >> 24 }; fwrite(p, 1, 4, f); }
static void put16(FILE *f, uint16_t v) { unsigned char p[2] = { v, v >> 8 }; fwrite(p, 1, 2, f); }

/* 16 kHz mono s16le PCM WAV into *out (malloc'd); samples or -1 */
static long read_wav(const char *path, int16_t **out)
{
    FILE *f = fopen(path, "rb"); unsigned char h[12], c[8];
    if (!f) { fprintf(stderr, "%s: cannot open\n", path); return -1; }
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) { fprintf(stderr, "%s: not a WAV file\n", path); fclose(f); return -1; }
    int fmt_ok = 0; long n = -1;
    while (fread(c, 1, 8, f) == 8) {
        uint32_t len = le32(c + 4);
        if (!memcmp(c, "fmt ", 4)) {
            unsigned char fm[16];
            if (len < 16 || fread(fm, 1, 16, f) != 16) break;
            if (le16(fm) != 1 || le16(fm + 2) != 1 || le32(fm + 4) != 16000 || le16(fm + 14) != 16) {
                fprintf(stderr, "%s: want 16 kHz mono 16-bit PCM (format %u, %u channels, %u Hz, %u bits)\n", path, le16(fm), le16(fm + 2), le32(fm + 4), le16(fm + 14));
                fclose(f); return -1;
            }
            fmt_ok = 1;
            if (len > 16) fseek(f, (long)(len - 16 + (len & 1)), SEEK_CUR);
        } else if (!memcmp(c, "data", 4)) {
            if (!fmt_ok) break;
            *out = malloc(len + 2);
            n = (long)(fread(*out, 1, len, f) / 2);
            break;
        } else fseek(f, (long)(len + (len & 1)), SEEK_CUR);
    }
    fclose(f);
    if (n < 0) fprintf(stderr, "%s: no fmt/data chunk\n", path);
    return n;
}

static int write_wav(const char *path, const int16_t *s, long n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "%s: cannot write\n", path); return -1; }
    fwrite("RIFF", 1, 4, f); put32(f, (uint32_t)(36 + n * 2)); fwrite("WAVEfmt ", 1, 8, f); put32(f, 16);
    put16(f, 1); put16(f, 1); put32(f, 16000); put32(f, 32000); put16(f, 2); put16(f, 16);
    fwrite("data", 1, 4, f); put32(f, (uint32_t)(n * 2));
    fwrite(s, 2, (size_t)n, f);
    return fclose(f);
}

int main(int argc, char **argv)
{
    float cap = 9, gain = 0; unsigned mode = 0; double warm = 0.5; int warm_front = 0; const char *vad = NULL; int o;
    while ((o = getopt(argc, argv, "c:sbw:Wv:g:")) != -1) switch (o) {
    case 'c': cap = (float)atof(optarg); break;
    case 's': mode |= MICDENOISE_SPEECH_SAFE; break;
    case 'b': mode |= MICDENOISE_BAND; break;
    case 'w': warm = atof(optarg); break;
    case 'W': warm_front = 1; break;
    case 'v': vad = optarg; break;
    case 'g': gain = (float)atof(optarg); break;
    default: fprintf(stderr, "usage: denoise-file [-c dB] [-s] [-b] [-w s] [-W] [-v file] [-g dB] in.wav out.wav\n"); return 2;
    }
    if (argc - optind != 2) { fprintf(stderr, "usage: denoise-file [-c dB] [-s] [-b] [-w s] [-W] [-v file] [-g dB] in.wav out.wav\n"); return 2; }
    int16_t *in; long n = read_wav(argv[optind], &in);
    if (n < 0) return 1;
    if (gain) { float g = powf(10.0f, -gain / 20.0f); for (long i = 0; i < n; i++) in[i] = (int16_t)lrintf(in[i] * g); }
    long lead = 0;                      /* the zeros in front: hassmic's lead-in, not through the reduction */
    while (lead < n && in[lead] == 0) lead++;
    if (lead == n) lead = 0;            /* all zeros: run it anyway */
    long body = n - lead, delay = mode & MICDENOISE_BAND ? MICDENOISE_DELAY_BAND : MICDENOISE_DELAY, flush = delay + MICDENOISE_FRAME;
    int16_t *out = calloc((size_t)n + flush + MICDENOISE_FRAME, sizeof *out), *tmp = calloc((size_t)body + flush + MICDENOISE_FRAME, sizeof *tmp);
    FILE *vf = vad ? fopen(vad, "w") : NULL;
    if (vad && !vf) { fprintf(stderr, "%s: cannot write\n", vad); return 1; }

    micdenoise_mode(mode);
    micdenoise_reset(cap);
    long w = (long)(warm * 16000);
    if (w > body) w = body;
    if (w > 0) micdenoise_run(in + (warm_front ? lead : n - w), (size_t)w, NULL);
    /* frame by frame, so the per-frame values can be logged; then zeros to flush the delay */
    long made = 0; static const int16_t zero[MICDENOISE_FRAME];
    for (long i = 0; i < body; i += MICDENOISE_FRAME) {
        size_t k = body - i < MICDENOISE_FRAME ? (size_t)(body - i) : MICDENOISE_FRAME;
        made += (long)micdenoise_run(in + lead + i, k, tmp + made);
        if (vf && k == MICDENOISE_FRAME) fprintf(vf, "%ld %.3f %.1f\n", i / MICDENOISE_FRAME, micdenoise_speech(), micdenoise_cap());
    }
    for (long i = 0; i < flush; i += MICDENOISE_FRAME) made += (long)micdenoise_run(zero, MICDENOISE_FRAME, tmp + made);
    if (made < body + delay) { fprintf(stderr, "short output: %ld of %ld\n", made, body + delay); return 1; }
    memcpy(out, in, (size_t)lead * sizeof *in);
    memcpy(out + lead, tmp + delay, (size_t)body * sizeof *tmp);
    if (gain) { float g = powf(10.0f, gain / 20.0f); for (long i = 0; i < n; i++) { float v = out[i] * g; out[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)lrintf(v); } }
    if (vf) fclose(vf);
    if (write_wav(argv[optind + 1], out, n)) return 1;
    fprintf(stderr, "%s: %ld samples, %ld zeros in front kept, cap %g dB%s%s, warm-up %.2f s %s -> %s\n", argv[optind], n, lead, cap,
            mode & MICDENOISE_SPEECH_SAFE ? ", speech-safe" : "", mode & MICDENOISE_BAND ? ", below 3 kHz only" : "",
            w / 16000.0, warm_front ? "from the start" : "from the end", argv[optind + 1]);
    return 0;
}
