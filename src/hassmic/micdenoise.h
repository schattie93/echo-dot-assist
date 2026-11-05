/* Noise reduction for the mic audio sent to the voice pipeline: RNNoise (src/third_party/rnnoise, v0.1.1), ahead of the
 * gain (micgain.h).  Not for the wake word, whose model is tuned to the stock stream.
 *
 * Why: even with the front end in listening mode a quietly spoken command sits only 6-8 dB over the noise floor of
 * micAsr (-65 dBFS; captures of 2026-09-30), and the gain lifts that floor with the speech to about -32 dBFS.  Amazon's
 * ASR path has no noise reduction of its own (AFE.cfg: the NoiseReductor belongs to the VoIP path).
 *
 * What it costs speech (issue #131, measured on 20 sentences of one speaker in one kitchen, Medium against Off, every
 * STT engine tried): RNNoise's per-band gains take the quiet consonants and everything over 3.5 kHz with the noise,
 * and exact sentences fell from 13 to 8 of 20 with the project's Whisper fine-tune, from 11 to 8 with Home Assistant
 * Cloud STT.  Hence the modes below, which keep the reduction away from speech. */
#ifndef MICDENOISE_H
#define MICDENOISE_H
#include <stddef.h>
#include <stdint.h>

#define MICDENOISE_FRAME 160            /* 10 ms at 16 kHz: what goes in comes out in blocks of this */

/* Modes (micdenoise_mode), any combination; 0 is the reduction as it was: the cap in every frame, in every band. */
#define MICDENOISE_SPEECH_SAFE 1        /* the cap only in frames RNNoise hears no speech in (its VAD); 1 dB while it does */
#define MICDENOISE_BAND 2               /* the reduction only below 3 kHz: the consonants above pass as they came */

/* The delay from input to output, in samples: the two resampling filters and RNNoise's frame, plus the crossover's in
 * MICDENOISE_BAND.  tests and tools line the two up by it. */
#define MICDENOISE_DELAY 192
#define MICDENOISE_DELAY_BAND 240

/* Which modes the next micdenoise_reset starts with.  Flags as above. */
void micdenoise_mode(unsigned flags);
/* A new stretch of audio starts: forget the last one.  max_db: the most the noise is taken down by (6, 9, 12). */
void micdenoise_reset(float max_db);
/* 16 kHz mono in, the same out, 12 ms later plus the wait for a full frame: returns the samples written to out, a multiple of MICDENOISE_FRAME that
 * can be 0; out must hold n + MICDENOISE_FRAME.  out == NULL: only listen (to settle on the room before a command). */
size_t micdenoise_run(const int16_t *in, size_t n, int16_t *out);
float micdenoise_speech(void);          /* RNNoise's speech probability of the last frame, 0..1 */
float micdenoise_cap(void);             /* the cap in force after the last frame, dB (moves in MICDENOISE_SPEECH_SAFE) */
#endif
