/* RNNoise on the 16 kHz mic stream: see micdenoise.h.
 *
 * RNNoise runs at 48 kHz in frames of 480, so every 160 samples go up by 3 (zero stuffing + low-pass), through
 * rnnoise_process_frame, and down by 3 again (the same low-pass, every third sample).  Its bands above 8 kHz stay empty,
 * which it copes with: the gains are per band.
 *
 * Level: RNNoise was trained on speech between full scale and 40 dB below it and ours arrives at -55 dBFS, but raising
 * the stream ahead of it (by 12, 24, 36 dB, and back down behind it) changed nothing measurable, so it gets it as it is.
 *
 * Left alone RNNoise takes the floor between words to digital silence (-65 to below -108 dBFS on the captures), and it
 * cuts into the weak sounds of quiet speech.  So the reduction is capped: the output is RNNoise's plus the input of the
 * same moment (RNNoise delays by one frame) at minus the cap.  By ear (user, a quiet sentence over the Echo's room
 * noise, 2026-09-30): artefacts at 18 dB, fine at 6, 9 and 12, which became the three strengths.
 *
 * Even capped at 9 dB it costs speech-to-text more than the noise did (issue #131): what RNNoise takes out of a frame
 * it takes per band, and its bands over 3.5 kHz and the quiet consonants go with the noise.  Two modes keep it off the
 * speech, each on its own or together (micdenoise_mode):
 * - MICDENOISE_SPEECH_SAFE: the cap follows RNNoise's own speech probability.  A frame it hears speech in gets 1 dB
 *   (SPEECH_CAP_DB), the first one at once (the probability belongs to the frame RNNoise is given, its output is the
 *   frame before: a frame of look-ahead, so the onset is kept), and the cap only climbs back after HOLD frames without
 *   speech, RELEASE_DB a frame: RNNoise's VAD sits low on fricatives and word ends, which must not be gated.  The mix
 *   ramps across each frame between the last keep and this one, so the cap never steps.
 * - MICDENOISE_BAND: only the difference RNNoise makes (its output minus the input of the same moment, capped as above)
 *   goes through the resampler down, then through a 3 kHz low-pass (XO_TAPS, Blackman: -6 dB at 3 kHz, through below
 *   2.5, gone above 3.5), and is added to the raw input delayed to match: below 3 kHz the reduction as before, above it
 *   the microphone's own signal, not even the resampler's 7 kHz low-pass.  The raw input lags the chain by a third of
 *   a sample (the two 96-tap filters delay 95 samples at 48 kHz) and the term added is the noise taken out, so the
 *   mismatch is in what is subtracted, in its top band only, not in the speech kept. */
#include "micdenoise.h"
#include <math.h>
#include <string.h>
#include "../third_party/rnnoise/rnnoise.h"

#define UP 3
#define F48 (MICDENOISE_FRAME * UP)
#define TAPS 96                         /* low-pass at 48 kHz: pass to 7 kHz, -60 dB from 9 kHz (Blackman) */
#define XO_TAPS 97                      /* crossover low-pass at 16 kHz, 3 kHz, odd: a delay of whole samples (48) */
#define XO_DELAY ((XO_TAPS - 1) / 2)
#define RAW_DELAY MICDENOISE_DELAY_BAND /* the raw input's lag to meet the crossover's output: 192 + 48 */

/* speech-safe (overridable with -D for trials with tools/denoise-file; the values: see the top of the file) */
#ifndef SPEECH_THR
#define SPEECH_THR 0.3f                 /* RNNoise's speech probability from which a frame counts as speech */
#endif
#ifndef SPEECH_CAP_DB
#define SPEECH_CAP_DB 1.0f              /* ... and its cap then: as good as off, the ramp in and out stays inaudible */
#endif
#ifndef HOLD
#define HOLD 15                         /* ... frames (150 ms) after the last speech frame before the cap climbs back */
#endif
#ifndef RELEASE_DB
#define RELEASE_DB 1.0f                 /* ... dB per frame it climbs: the full Medium cap back within 80 ms more */
#endif

static float lp[TAPS];
static float up_hist[TAPS / UP - 1];    /* the last input samples: the zero-stuffed signal only has every third */
static float down_hist[TAPS - 1];
static float dry[F48];                  /* the frame RNNoise was given before this one: what its output belongs to */
static int16_t pend[MICDENOISE_FRAME]; static size_t npend;
static DenoiseState *st;
static float speech, keep;              /* keep: how much of the input stays, 10^(-cap / 20) */
static unsigned mode, mode_next;        /* MICDENOISE_* flags in force, and for the next reset */
static float max_db, cap_cur, keep_prev; static int hold;   /* speech-safe: the cap asked for, the one in force, the last frame's keep, frames of hold left */
static float xo[XO_TAPS], xo_hist[XO_TAPS - 1]; static int16_t raw_hist[RAW_DELAY];   /* band: crossover and the raw input's delay line */

static void init(void)
{
    double sum = 0;
    for (int i = 0; i < TAPS; i++) {
        double t = i - (TAPS - 1) / 2.0, x = M_PI * t / UP;                     /* cutoff 8 kHz = fs / (2 * UP) */
        double w = 0.42 - 0.5 * cos(2 * M_PI * i / (TAPS - 1)) + 0.08 * cos(4 * M_PI * i / (TAPS - 1));
        lp[i] = (float)((fabs(t) < 1e-9 ? 1.0 : sin(x) / x) * w);
        sum += lp[i];
    }
    for (int i = 0; i < TAPS; i++) lp[i] = (float)(lp[i] / sum);
    sum = 0;
    for (int i = 0; i < XO_TAPS; i++) {
        double t = i - XO_DELAY, x = M_PI * t * 3000.0 / 8000.0;                /* cutoff 3 kHz of 16 kHz */
        double w = 0.42 - 0.5 * cos(2 * M_PI * i / (XO_TAPS - 1)) + 0.08 * cos(4 * M_PI * i / (XO_TAPS - 1));
        xo[i] = (float)((t == 0 ? 1.0 : sin(x) / x) * w);
        sum += xo[i];
    }
    for (int i = 0; i < XO_TAPS; i++) xo[i] = (float)(xo[i] / sum);
}

void micdenoise_mode(unsigned flags) { mode_next = flags & (MICDENOISE_SPEECH_SAFE | MICDENOISE_BAND); }

void micdenoise_reset(float db)
{
    max_db = db; keep = powf(10.0f, -db / 20.0f);
    mode = mode_next; cap_cur = db; keep_prev = keep; hold = 0;
    if (!lp[TAPS / 2]) init();
    if (st) rnnoise_destroy(st);        /* rnnoise_init on a used state would leak its three buffers */
    st = rnnoise_create(NULL);
    memset(up_hist, 0, sizeof up_hist); memset(down_hist, 0, sizeof down_hist); memset(dry, 0, sizeof dry);
    memset(xo_hist, 0, sizeof xo_hist); memset(raw_hist, 0, sizeof raw_hist);
    npend = 0; speech = 0;
}

float micdenoise_speech(void) { return speech; }
float micdenoise_cap(void) { return cap_cur; }

/* speech-safe: the cap for the frame about to come out, from this frame's speech probability */
static float cap_follow(float p)
{
    if (p >= SPEECH_THR) { hold = HOLD; return SPEECH_CAP_DB; }
    if (hold > 0) { hold--; return SPEECH_CAP_DB; }
    float c = cap_cur + RELEASE_DB;
    return c > max_db ? max_db : c;
}

static void frame(const int16_t *in, int16_t *out)
{
    enum { H = TAPS / UP - 1 };
    float x[H + MICDENOISE_FRAME], a[F48], b[TAPS - 1 + F48];
    memcpy(x, up_hist, sizeof up_hist);
    for (int i = 0; i < MICDENOISE_FRAME; i++) x[H + i] = in[i];
    memcpy(up_hist, x + MICDENOISE_FRAME, sizeof up_hist);
    /* up: output sample 3 i + p is the sum over the input samples that fall under the filter, taps p, p + 3, ... */
    for (int i = 0; i < MICDENOISE_FRAME; i++)
        for (int p = 0; p < UP; p++) {
            float s = 0;
            for (int k = 0; k <= H; k++) s += lp[p + UP * k] * x[H + i - k];
            a[UP * i + p] = s * UP;
        }
    memcpy(b, down_hist, sizeof down_hist);
    speech = rnnoise_process_frame(st, b + TAPS - 1, a);
    if (mode & MICDENOISE_SPEECH_SAFE) {
        cap_cur = cap_follow(speech);
        float k1 = powf(10.0f, -cap_cur / 20.0f), k0 = keep_prev;
        for (int i = 0; i < F48; i++) {
            float k = k0 + (k1 - k0) * (i + 1) / F48;
            b[TAPS - 1 + i] = b[TAPS - 1 + i] * (1 - k) + dry[i] * k;
        }
        keep_prev = k1;
    } else
        for (int i = 0; i < F48; i++) b[TAPS - 1 + i] = b[TAPS - 1 + i] * (1 - keep) + dry[i] * keep;
    if (mode & MICDENOISE_BAND) for (int i = 0; i < F48; i++) b[TAPS - 1 + i] -= dry[i];   /* only what the reduction changed */
    memcpy(dry, a, sizeof dry);
    memcpy(down_hist, b + F48, sizeof down_hist);
    if (!out && !(mode & MICDENOISE_BAND)) return;   /* band keeps its delay lines in step even while only listening */
    float y[MICDENOISE_FRAME];
    for (int i = 0; i < MICDENOISE_FRAME; i++) {
        float s = 0;
        for (int k = 0; k < TAPS; k++) s += lp[k] * b[UP * i + TAPS - 1 - k];
        y[i] = s;
    }
    if (mode & MICDENOISE_BAND) {
        /* the difference below 3 kHz on top of the raw input of the same moment (RAW_DELAY back) */
        float z[XO_TAPS - 1 + MICDENOISE_FRAME];
        memcpy(z, xo_hist, sizeof xo_hist); memcpy(z + XO_TAPS - 1, y, sizeof y);
        memcpy(xo_hist, z + MICDENOISE_FRAME, sizeof xo_hist);
        for (int i = 0; i < MICDENOISE_FRAME; i++) {
            float s = 0;
            for (int k = 0; k < XO_TAPS; k++) s += xo[k] * z[i + XO_TAPS - 1 - k];
            y[i] = raw_hist[i] + s;
        }
        memmove(raw_hist, raw_hist + MICDENOISE_FRAME, sizeof raw_hist - MICDENOISE_FRAME * sizeof raw_hist[0]);
        memcpy(raw_hist + RAW_DELAY - MICDENOISE_FRAME, in, MICDENOISE_FRAME * sizeof in[0]);
        if (!out) return;
    }
    for (int i = 0; i < MICDENOISE_FRAME; i++) {
        float s = y[i];
        out[i] = s > 32767 ? 32767 : s < -32768 ? -32768 : (int16_t)lrintf(s);
    }
}

size_t micdenoise_run(const int16_t *in, size_t n, int16_t *out)
{
    size_t made = 0;
    if (!st) micdenoise_reset(12);
    while (n) {
        size_t k = MICDENOISE_FRAME - npend; if (k > n) k = n;
        memcpy(pend + npend, in, k * sizeof *in);
        npend += k; in += k; n -= k;
        if (npend < MICDENOISE_FRAME) break;
        frame(pend, out ? out + made : NULL);
        made += MICDENOISE_FRAME; npend = 0;
    }
    return out ? made : 0;
}
