/* Noise reduction of the pipeline's mic audio (micdenoise.c) on espeak speech brought down to the level and the noise
 * floor the Echo delivers: speech at -57 dBFS over white noise at -65 dBFS, as in the captures of 2026-09-30.
 * Then the two modes of issue #131 on the same file: speech-safe keeps the cap off the speech (its top band most of
 * all) and still takes the floor down after it; band leaves everything over 3 kHz as it came.
 * usage: micdenoise_test testdata/alexa_espeak.raw */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "micdenoise.h"

#define RATE 16000
#define MAXN (RATE * 10)
static int bad;
static void check(int ok, const char *what, double v) { printf("%s %s: %.1f\n", ok ? "ok  " : "FAIL", what, v); if (!ok) bad = 1; }

static double level(const int16_t *s, size_t a, size_t b)
{
    double sum = 0;
    for (size_t i = a; i < b; i++) sum += (double)s[i] * s[i];
    return 10 * log10(sum / (b - a) / (32768.0 * 32768.0) + 1e-12);
}

/* the level of [a, b) above or below a cutoff: a 2nd-order Butterworth high- or low-pass (two passes of a biquad), enough
 * to tell the bands of white noise and of speech apart */
static double band_level_in(const int16_t *s, size_t a, size_t b, double cutoff_hz, int high, const int16_t *speech, size_t speech_off);
static double band_level(const int16_t *s, size_t a, size_t b, double cutoff_hz, int high) { return band_level_in(s, a, b, cutoff_hz, high, NULL, 0); }
/* speech != NULL: only the 10 ms frames in which that (clean) signal, speech_off samples ahead of s, is over -60 dBFS,
 * i.e. inside the phrases and not the pauses between them, where the floor alone would decide */
static double band_level_in(const int16_t *s, size_t a, size_t b, double cutoff_hz, int high, const int16_t *speech, size_t speech_off)
{
    double w = tan(M_PI * cutoff_hz / RATE), k = w * w, q = sqrt(2) * w, n = 1 / (1 + q + k);
    double b0 = high ? n : k * n, b1 = high ? -2 * n : 2 * k * n, b2 = b0, a1 = 2 * (k - 1) * n, a2 = (1 - q + k) * n;
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0, sum = 0; size_t from = a > RATE / 4 ? a - RATE / 4 : 0, cnt = 0;   /* settle for 250 ms first */
    for (size_t i = from; i < b; i++) {
        double x = s[i], y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x; y2 = y1; y1 = y;
        if (i < a) continue;
        if (speech) {
            size_t f = (i - speech_off) / 160 * 160; double e = 0;
            for (size_t j = f; j < f + 160; j++) e += (double)speech[j] * speech[j];
            if (10 * log10(e / 160 / (32768.0 * 32768.0) + 1e-12) < -60) continue;
        }
        sum += y * y; cnt++;
    }
    return 10 * log10(sum / (cnt ? cnt : 1) / (32768.0 * 32768.0) + 1e-12);
}

static int16_t clean[MAXN], in[MAXN], out[MAXN + MICDENOISE_FRAME];

int main(int argc, char **argv)
{
    FILE *f = fopen(argc > 1 ? argv[1] : "testdata/alexa_espeak.raw", "rb");
    if (!f) { printf("FAIL cannot open the speech file\n"); return 1; }
    size_t n = fread(clean, 2, MAXN, f), made = 0;
    fclose(f);
    /* the file: silence to 1.5 s, four phrases to 6.0 s (-21 dBFS), silence.  Speech to -57 dBFS, noise at -65 dBFS */
    double g = pow(10, (-57 - level(clean, RATE * 3 / 2, RATE * 6)) / 20), na = 32768 * pow(10, -65 / 20.0) * sqrt(3);
    unsigned seed = 1;
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        in[i] = (int16_t)lrint(clean[i] * g + na * (((seed >> 8) & 0xffff) / 32768.0 - 1.0));
        clean[i] = (int16_t)lrint(clean[i] * g);
    }
    micdenoise_reset(12);
    for (size_t i = 0; i < n; i += 250) made += micdenoise_run(in + i, n - i < 250 ? n - i : 250, out + made);   /* odd blocks */
    check(made == n / MICDENOISE_FRAME * MICDENOISE_FRAME, "whole frames out, none lost", (double)made);

    enum { D = 192 };                     /* delay: 12 ms */
    /* from cold RNNoise needs a second or two to settle on white noise (0.2 s on the Echo's room noise): hassmic lets
     * it hear the second before the command first */
    double noise_in = level(in, RATE, RATE * 3 / 2), noise_out = level(out, RATE + D, RATE * 3 / 2 + D);
    check(noise_in - noise_out > 5 && noise_in - noise_out < 12.5, "noise 1 to 1.5 s after a cold start, dB down", noise_in - noise_out);
    double tail = level(in, RATE * 7, RATE * 8) - level(out, RATE * 7 + D, RATE * 8 + D);
    check(tail > 10 && tail < 12.5, "noise after the speech: down by the cap of 12 dB, no more", tail);
    double sp_clean = level(clean, RATE * 3 / 2, RATE * 6), sp_out = level(out, RATE * 3 / 2 + D, RATE * 6 + D);
    check(fabs(sp_out - sp_clean) < 3, "speech keeps its level, dB off the clean speech", sp_out - sp_clean);
    double err = 0, ref = 0;              /* what is left besides the clean speech, over the phrases */
    for (size_t i = RATE * 3 / 2; i < (size_t)RATE * 6; i++) { double e = out[i + D] - clean[i]; err += e * e; ref += (double)clean[i] * clean[i]; }
    double in_err = 0;
    for (size_t i = RATE * 3 / 2; i < (size_t)RATE * 6; i++) { double e = in[i] - clean[i]; in_err += e * e; }
    check(10 * log10(ref / err) > 10 * log10(ref / in_err) + 2, "speech to everything else better than before, dB", 10 * log10(ref / err) - 10 * log10(ref / in_err));
    check(micdenoise_speech() < 0.5, "no speech seen in the noise at the end, probability x 100", micdenoise_speech() * 100);

    micdenoise_reset(6);                  /* the lowest strength */
    made = 0;
    for (size_t i = 0; i < n; i += 250) made += micdenoise_run(in + i, n - i < 250 ? n - i : 250, out + made);
    tail = level(in, RATE * 7, RATE * 8) - level(out, RATE * 7 + D, RATE * 8 + D);
    check(tail > 5 && tail < 6.5, "cap of 6 dB", tail);

    micdenoise_reset(12);                 /* warming up on the room (out == NULL) gives nothing back */
    check(micdenoise_run(in, RATE, NULL) == 0, "listening only returns no samples", 0);

    /* speech-safe at the Medium cap: the speech's top band stays, the cap drops to 1 dB inside the speech and is back on
     * the floor after it; the whole-file mix no longer costs the speech its consonants (4-8 kHz: espeak's are there) */
    double sp_hi_in = band_level_in(in, RATE * 3 / 2, RATE * 6, 4000, 1, clean, 0), sp_hi_plain, sp_hi_safe, cap_in_speech = 0, cap_after = 0;
    micdenoise_mode(0); micdenoise_reset(9); made = 0;
    for (size_t i = 0; i < n; i += 250) made += micdenoise_run(in + i, n - i < 250 ? n - i : 250, out + made);
    sp_hi_plain = band_level_in(out, RATE * 3 / 2 + D, RATE * 6 + D, 4000, 1, clean, D);
    micdenoise_mode(MICDENOISE_SPEECH_SAFE); micdenoise_reset(9); made = 0;
    for (size_t i = 0; i < n; i += MICDENOISE_FRAME) {
        made += micdenoise_run(in + i, n - i < MICDENOISE_FRAME ? n - i : MICDENOISE_FRAME, out + made);
        if (i == RATE * 3) cap_in_speech = micdenoise_cap();          /* the second phrase */
        if (i == RATE * 15 / 2) cap_after = micdenoise_cap();         /* 1.5 s of noise after the last one */
    }
    sp_hi_safe = band_level_in(out, RATE * 3 / 2 + D, RATE * 6 + D, 4000, 1, clean, D);
    check(sp_hi_in - sp_hi_plain > 2, "plain Medium takes the speech's 4-8 kHz down, dB (what issue #131 measured)", sp_hi_in - sp_hi_plain);
    check(fabs(sp_hi_in - sp_hi_safe) < 1.5, "speech-safe keeps the speech's 4-8 kHz, dB off the input", sp_hi_safe - sp_hi_in);
    check(cap_in_speech <= 1.01, "speech-safe: cap inside a phrase, dB", cap_in_speech);
    check(cap_after > 8.9, "speech-safe: cap back on the floor 1.5 s after the speech, dB", cap_after);
    tail = level(in, RATE * 7, RATE * 8) - level(out, RATE * 7 + D, RATE * 8 + D);
    check(tail > 7.5 && tail < 9.5, "speech-safe: noise after the speech still down by the cap, dB", tail);
    check(micdenoise_run(in, RATE, NULL) == 0, "speech-safe: listening only returns no samples", 0);

    /* band: over 3 kHz the output is the input (noise and speech alike), below it the floor is down by the cap */
    enum { DB = MICDENOISE_DELAY_BAND };
    micdenoise_mode(MICDENOISE_BAND); micdenoise_reset(9); made = 0;
    for (size_t i = 0; i < n; i += 250) made += micdenoise_run(in + i, n - i < 250 ? n - i : 250, out + made);
    check(made == n / MICDENOISE_FRAME * MICDENOISE_FRAME, "band: whole frames out", (double)made);
    double hi_out = band_level(out, RATE * 7 + DB, RATE * 8 + DB, 4000, 1), hi_in = band_level(in, RATE * 7, RATE * 8, 4000, 1);
    check(fabs(hi_out - hi_in) < 0.3, "band: noise over 4 kHz untouched, dB off the input", hi_out - hi_in);
    double lo_in = band_level(in, RATE * 7, RATE * 8, 2000, 0), lo_out = band_level(out, RATE * 7 + DB, RATE * 8 + DB, 2000, 0);
    check(lo_in - lo_out > 7 && lo_in - lo_out < 9.5, "band: noise under 2 kHz down by the cap, dB", lo_in - lo_out);
    double sp_hi_band = band_level_in(out, RATE * 3 / 2 + DB, RATE * 6 + DB, 4000, 1, clean, DB);
    check(fabs(sp_hi_band - sp_hi_in) < 0.3, "band: speech over 4 kHz untouched, dB off the input", sp_hi_band - sp_hi_in);
    err = 0;                              /* and lined up with the input: the delay is MICDENOISE_DELAY_BAND */
    for (size_t i = RATE * 3 / 2; i < (size_t)RATE * 6; i++) { double e = out[i + DB] - clean[i]; err += e * e; }
    check(10 * log10(ref / err) > 10 * log10(ref / in_err) + 0.5, "band: speech to everything else better than before, dB (white noise: only its third below 3 kHz goes)", 10 * log10(ref / err) - 10 * log10(ref / in_err));
    check(micdenoise_run(in, RATE, NULL) == 0, "band: listening only returns no samples", 0);
    micdenoise_mode(0);
    return bad;
}
