# Noise reduction (RNNoise) degrades STT accuracy – measured

**Device / setup**
- Echo Dot 2 (2016), fork `schattie93/echo-dot-assist`, Home Assistant Assist pipeline, "Wake word in the request" on.
- STT under test: our own Whisper fine-tune (large-v3-turbo, ACFT + LoRA, served by whisper.cpp with short audio context), plus two unmodified controls: `primeline/whisper-large-v3-turbo-german` (whisper.cpp) and `faster-whisper large-v3` (int8). Amazon's cloud STT (HA Cloud) is shown as a third reference where we have it.
- Noise reduction in the app was set to **Medium** (RNNoise, 9 dB cap) from 02 Nov; **Off** from 05 Nov 14:29. Mic level unchanged throughout. Same kitchen, same Dot position, same speakers.

This is a report of measurements, not a request. The STT effect of the RNNoise stage is listed as unmeasured in `PLAN.md` ("Not done: what STT makes of it (the reason to have it)", `PLAN.md:1012`; "Not done: a case where STT fails without noise reduction and succeeds with it", `PLAN.md:1318`) – below is that measurement, for our room and speakers.

## A. Dot vs. laptop microphone, same 40 sentences (Medium)

40 sentences that were never in our training set (2 speakers × 20: household commands, timers, general questions), each spoken once into the Dot (NR Medium) and once into a laptop microphone in a quiet room, same speakers. Exact-match sentence accuracy (wake word stripped, punctuation/case ignored):

| STT | Dot, NR Medium | Laptop mic |
|---|---|---|
| our fine-tune (turbo v3) | **13/40** | **37/40** |
| faster-whisper large-v3 (unmodified) | 8/40 | 33/40 |

The fine-tune is trained on Dot audio and still loses 24 sentences to the chain; the unmodified large-v3 loses 25. So it is not a model/domain problem.

## B. What the signal looks like (40 pairs, `signal_vergleich.py`)

Medians, Dot (NR Medium) / laptop:

- Noise floor −59 / −47 dBFS, speech −25 / −30 dBFS – i.e. the Dot has the *better* SNR on paper (33 vs 17 dB).
- Speech energy 4–6 kHz relative to 300–1000 Hz: 10–15 dB lower on the Dot.
- Consonant bursts above 5.5 kHz detectable in **2 of 40** Dot files vs **76 events** in the laptop files.
- No measurable reverb difference (decay after speech end 0.05 / 0.04 s), no clipping on either side.
- Spectrograms of the Dot files show smeared harmonics and a "musical noise" texture after the wake word – the signature of a per-band spectral gate, not of a room or a level problem.

Spectrogram, one sentence ("Alexa, what is the capital of Portugal?", speaker C, sentence 16), top to bottom: Dot with NR Medium, Dot with NR Off, laptop mic:

![spectrogram](https://raw.githubusercontent.com/schattie93/echo-dot-assist/docs/rnnoise-stt-measurement/docs/measurements/rnnoise-stt/spektrogramm_c16_medium_off_laptop.png)

With Medium there is essentially nothing above 3 kHz after the wake word; with Off the consonant bursts up to 7 kHz are back.

## C. Medium → Off, same speaker, same 20 sentences, same room

Speaker C repeated his 20 sentences three hours later with noise reduction Off; nothing else changed.

| STT | NR Medium | NR Off |
|---|---|---|
| our fine-tune, turbo v2 | 4/20 | **9/20** |
| our fine-tune, turbo v3 | 8/20 | **11/20** |
| our fine-tune, turbo v4 | 8/20 | **13/20** |
| primeline turbo-german (unmodified) | 3/20 | 5/20 |
| Amazon cloud STT (HA Cloud) | 8/20 | 11/20 |

Every recognizer gains, including Amazon's, which rules out our fine-tune as the cause. Signal with Off: speech spectrum above 3.5 kHz +8 dB (now level with the laptop), noise floor −48 dBFS (the −59 dBFS with Medium was the gate, not real quiet), speech level unchanged (−25 dBFS). The remaining gap to the laptop (13/20 vs 17/20 for v4) is mostly long free-form questions.

Per-sentence outputs for all models and both settings, the per-file signal metrics and the measurement script are in my fork: [docs/measurements/rnnoise-stt](https://github.com/schattie93/echo-dot-assist/tree/docs/rnnoise-stt-measurement/docs/measurements/rnnoise-stt) (`sternchen_off_ergebnisse.csv`, `signal_vergleich.csv`, `signal_vergleich_off.csv`, `signal_vergleich.py`; mean speech spectra Medium/Off/laptop in `spektrum_off.png`).

## Suggestion (take or leave)

- Consider **Off** (or Low) as the default, and a note in the docs that Medium/High removes the 4–8 kHz band where consonants live – for Whisper-class STT that costs more than the noise gains. For our kitchen the 9 dB cap on Medium was enough to halve accuracy.
- If useful, the measurement script (`signal_vergleich.py`, numpy/scipy, ~200 lines: SNR, band energies, burst count, decay, lead/tail per file) can be contributed under the repo's license.

## Two more observations, no proposal attached

- **micRaw vs micAsr** (`scripts/mic-compare.sh -l 10`, one sentence): correlation 0.96; micAsr falls relative to micRaw over the sentence, ≈ −4 dB mid-sentence, ≈ −2 dB at the end. So the level droop after the wake word that we see in the recordings originates in Amazon's front end before hassmic, not in `micgain.c`. We are not patching the AGC because of this.
- **Aligner spread** (02 Nov, with "Wake word in the request" off): the reported word-end position scattered from −90 ms to +380 ms relative to the actual end of "Alexa", which in the worst cases clipped the start of the command. With "Wake word in the request" on this no longer matters for us, so just a data point.

Happy to share the raw WAV pairs privately if that helps.
