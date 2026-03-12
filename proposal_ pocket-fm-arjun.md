

Improving Long-Form Expressive Narrative TTS for Hindi
Arjun BahugunaKiran RajaPranav AnuraagWilliam Xing
## 05 March 2026
## 1  Abstract
Long-form expressive Hindi text-to-speech (TTS) remains difficult even when sentence-level qual-
ity  is  strong.   This  proposal  focuses  on  audiobook-style  synthesis  with  IndicParler-TTS  as  the
primary baseline, supported by comparisons to other large-scale systems with Hindi or Malayalam
coverage.  The core gap we address is one essential for the audiobook generation task:  narrative
consistency across long contexts.  This entails preserving speaker identity, prosodic evolution, and
style adherence over chapters rather than isolated sentences.  Our methodology combines controlled
baseline generation, downstream stylization, and long-context conditioning experiments, evaluated
with objective and human-centered metrics.  The goal is a reproducible Hindi-focused method and
evaluation framework for diagnosing and reducing degradations in long-form, expressive narrative
## TTS.
## 2  Introduction
The recent generation of Indic TTS systems has materially improved language coverage, control-
lability, and voice quality.  IndicVoices-R provides a large multilingual Indian speech resource with
broad  language  and  speaker  diversity,  including  Hindi  and  Malayalam[10].   IndicParler-TTS  ex-
tends the Parler-TTS paradigm to Indic languages and introduces natural-language style-caption
control with a dual-tokenizer setup (transcript tokenizer and style-caption tokenizer), enabling di-
rect prompt-based control of traits such as speaking rate, pitch, and recording conditions [3, 9].
However,  we observe that long-form generation,  speaker consistency,  and expressive narration is
challenging for TTS models trained on English speech data.  We hypothethize the problem is even
starker for Hindi,  due to lack of large-scale data and pre-trained model support.  To tackle this
problem for long-form expressive narration, we model generation as conditional sequence synthesis:
y
## 1:T
∼ p
θ
## (y
## 1:T
| x
## 1:N
## ,s),
where x
## 1:N
is the Hindi transcript, s is a style-caption prompt, and y
## 1:T
is generated speech.  This
formulation  highlights  the  main  research  tension:  a  model  may  satisfy  local  prompt  constraints
while still failing to maintain global narrative state over extended duration.
## 2.1  Model Landscape
IndicParler-TTS (AI4Bharat):  An  autoregressive,  instruction-based  model  built  on  Parler-
TTS, trained on over 1,800 hours of Indic speech [3, 9].  Supported languages include Hindi
## 1

and Malayalam.  It utilizes a two-tokenizer conditioning system for transcripts and style cap-
tions.   We  designate  this  as  our  primary  baseline  due  to  its  open-source  nature  and  high
controllability.
IndicF5 (AI4Bharat):  A  non-autoregressive  system  derived  from  the  F5-TTS  flow-matching
family  with  a  Diffusion  Transformer  backbone  [6,  4].   It  supports  11  Indian  languages,  in-
cluding  Hindi  and  Malayalam,  via  a  reference-audio-guided  pipeline.   We  use  this  to  test
whether non-autoregressive generation can maintain long-range narrative coherence.
svara-TTS v1 (Kenpath/Orpheus-style):  Reports support for 19 languages and utilizes lightweight
end-of-utterance emotion tags [11].  While useful for short-form expressivity, we hypothesize
that its discrete tag-based conditioning may struggle with the continuous transitions required
for multi-paragraph narrative consistency.
Bulbul v3 / Saaras API (Sarvam AI):  A production-focused system supporting 11 languages
and  30+  voices  with  low-latency  deployment  [1,  2].   As  a  closed-source  API,  it  serves  as
a production reference point but is excluded from latent-level architectural interventions in
## Phase 3.
SeamlessExpressive (Meta):  A prosody-preserving system focused on expressive speech-to-speech
translation [7, 5].  Due to its lack of native Hindi/Malayalam support, it serves exclusively as
an architectural reference for our prosody transfer strategies rather than a direct synthesis
baseline.
## 3  Methodology
Our pipeline has three connected phases with a single fixed Hindi narrative test set.
PhaseTechnical objectives and execution
Baseline and DiagnosisFix  a  Hindi  narrative  subset  and  generate  baseline  audio  with
IndicParler-TTS   under   consistent   transcript   and   style-caption
prompts [3].  Archive outputs and prompts to enable exact replays,
then  compute  baseline  intelligibility  and  consistency  metrics  as  the
reference profile for subsequent phases.
Downstream Narrative StylizerTrain an audio-to-audio stylizer on Hindi expressive material and ap-
ply it to frozen baseline outputs to modulate narrative delivery with-
out changing text inputs.  Compare stylized versus baseline outputs
on identical passages to isolate stylizer impact on long-form consis-
tency.
Long-Context Architecture AnalysisDerive candidate long-context conditioning strategies for IndicParler-
TTS by extending style-caption control from sentence scope to chap-
ter scope.  Apply each candidate to fixed Hindi passages and retain
only variants that improve consistency while preserving intelligibility
and prompt adherence [3, 9].
Table 1:  Methodology phases and technical outputs.
Phase 1: Baseline: We run IndicParler-TTS with fixed transcript segmentation, style-caption
templates, and decoding settings, then archive outputs and prompts for exact replay [3].  Baseline
metrics establish reference intelligibility, identity stability, and style adherence.
## 2

Phase 2: Narrative stylization:  We add an audio-to-audio stylizer on top of frozen baseline
outputs to improve continuity without changing textual content.  This isolates whether long-form
gains can be obtained from post-generation refinement alone.
Phase 3: Improving long-context performance:  We extend conditioning from sentence-
local  prompts  to  chapter-level  state  variables,  testing  whether  the  model  can  preserve  narrative
trajectory over longer windows.  Candidate mechanisms include temporal adapters and low-rank
updates on conditioning pathways.
## 4  Evaluation Benchmarks
Evaluation relies on a dedicated long-form Hindi test suite and human listening analysis inspired
by long-form TTS evaluation practice [8].  We report intelligibility (ASR-based error rates), speaker
consistency across segments (embedding drift), and prompt-following consistency for style captions.
To  improve  diagnostic  power  for  long-form  Hindi  narration,  we  include  explicit  measurement  of
speaker identity drift,  classifier-based verification of expressive attributes,  and F
## 0
-based prosody
tracking across segment boundaries.
4.1  Summary of Evaluation Metrics
- Intelligibility & Accuracy:  Measured via ASR-based Word Error Rate (WER) and se-
mantic alignment (L
sem
## ).
- Speaker Identity Stability:  Quantified by the speaker-drift metric D
spk
over sequential
segments.
- Prosodic Continuity:  Assessed through inter-segment pitch statistics (L
prosody
) and dy-
namic style smoothness (L
dyn
## ).
- Style Adherence:  Evaluated via attribute prediction error (L
attr
) for controlled factors like
rate and pitch.
- Qualitative Assessment:   Human-centered  evaluation  focusing  on  listener  fatigue,  per-
ceived continuity, and pairwise chapter-level preference tests.
## 4.2  Evaluation Methodology
Evaluation  combines  automatic  and  human  assessment.   Automatic  metrics  include  ASR-based
error  rates,  embedding  drift,  and  style-prompt  adherence.   Human  assessment  follows  long-form
voice-quality practice, including perceived continuity and listener fatigue dimensions [8].  We also
perform pairwise chapter-level preference tests between baseline and each intervention.
Data protocol is centered on IndicVoices-R [10].  We use speaker-disjoint splits for model fitting
and validation, and construct two Hindi-focused evaluation subsets:  (i) a style-controlled set with
repeated transcripts and varied style captions for direct estimation of L
style
, and (ii) a long-form
chapter set with fixed segmentation for drift and continuity analysis.  This separation ensures that
short-form prompt adherence and long-form narrative stability are evaluated independently.
## 3

To make phase comparisons quantitative, we define speaker-drift and consistency objectives over
segment embeddings e
k
## :
## D
spk
## =
## 1
## K− 1
## K−1
## X
k=1
## 
## 1−
e
## ⊤
k
e
k+1
## ∥e
k
## ∥
## 2
## ∥e
k+1
## ∥
## 2
## 
## .
## Lower D
spk
indicates  stronger  identity  stability.   We  also  aggregate  prosodic  smoothness  with
segment-level pitch statistics f
k
## :
## L
prosody
## =
## 1
## K− 1
## K−1
## X
k=1
## ∥f
k+1
− f
k
## ∥
## 2
## .
Our optimization target for model selection is a weighted criterion
J = λ
## 1
WER + λ
## 2
## D
spk
+ λ
## 3
## L
prosody
+ λ
## 4
## L
style
## ,
where L
style
measures mismatch between intended and predicted expressive attributes.  We instan-
tiate this term as
## L
style
= αL
sem
+ β L
attr
+ γ L
dyn
## ,
where L
sem
is semantic alignment between the style caption and generated audio style embedding,
## L
attr
is attribute prediction error for controllable factors, and L
dyn
is inter-segment style smooth-
ness to penalize abrupt chapter-level shifts.
The  expected  outcome  is  not  merely  higher  sentence  quality,  but  improved  paragraph-level  and
chapter-level coherence and expressiveness for Hindi narration.  Concretely, we expect reductions
in D
spk
and L
prosody
at comparable intelligibility, indicating that conditioning improvements can
reduce long-form drift without collapsing expressivity.
## 5  Limitations
Three risks remain.  First, objective metrics only partially capture narrative quality; human evalu-
ation is still necessary.  Second, API-only systems (e.g., Bulbul) limit architectural interpretability
and  intervention  depth.   Third,  model-card  language  support  does  not  guarantee  equal  quality
across all dialectal or domain conditions; careful Hindi-domain test design is required.
## 6  Conclusion
This proposal refocuses long-form expressive TTS from a general multilingual framing to a Hindi-
first research program.  By centering IndicParler-TTS, incorporating Hindi-compatible comparator
models, and enforcing chapter-level evaluation criteria, the work targets a practically relevant gap
for audiobook-scale synthesis:  maintaining identity and expressive flow over long contexts, not just
sentence-level naturalness.
## 4

## 7  Bibliography
## References
[1]  Sarvam AI. Bulbul v3 api documentation. Official documentation, 2025. URL https://docs.
sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/overview.
[2]  Sarvam AI.  Bulbul v3 product page.  Official website, 2025.  URL https://www.sarvam.ai/
blog/introducing-bulbul-v3.
[3]  AI4Bharat.  Indicparler-tts.  Hugging Face model card,  2024.  URL https://huggingface.
co/ai4bharat/indic-parler-tts.
[4]  AI4Bharat. Indicf5. GitHub repository and model release, 2025. URL https://github.com/
AI4Bharat/IndicF5.
[5]  Loic Barrault, Yu-An Chung, Chung-Cheng Chiu, et al.  Seamlessm4t:  Massively multilingual
and multimodal machine translation, 2023.  URL https://arxiv.org/abs/2312.05187.
[6]  Yushun Chen, Zhifeng Ge, Yifu Wang, et al. F5-tts:  A fairytaler that fakes fluent and faithful
speech with flow matching, 2024.  URL https://arxiv.org/abs/2410.06885.
[7]  Meta  FAIR.Seamlessexpressive.Hugging  Face  model  card,   2024.URL https://
huggingface.co/facebook/seamless-expressive.
[8]  Maria   Jimenez,   George   Fazekas,   Masataka   Goto,   Mark   Gales,   Panayiotis   Georgiou,
Seetha  Nair,   Anupriya  Ramakrishnan,   Shunsuke  Sato,   Karan  Singh,   and  Rafael  Valle.
Choice  of  voices:    A  large-scale  evaluation  of  text-to-speech  voice  quality  for  long-form
content.In Proceedings of the 2020 CHI Conference on Human Factors in Comput-
ing  Systems,   2020.URL  https://www.mozillafoundation.org/research/library/
choice-of-voices-a-large-scale-evaluation-of-text-to-speech-voice-quality-for-long-form-content/.
[9]  Dan  Lyth  and  Simon  King.   Natural  language  guidance  of  high-fidelity  text-to-speech  with
synthetic annotations, 2024.  URL https://arxiv.org/abs/2402.01912.
[10]  Dhanunjaya Kumar Nirmala, Laurent Shafey, S. Anoop, Mitesh M. Kumar, Savitha Abraham,
et al.  Indicvoices-r:  A large multilingual indic speech corpus for speech generation tasks.  In
NeurIPS 2024 Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?
id=Czvnfc1i7c.
[11]  Kenpath  Technologies.svara-tts  v1.Hugging  Face  model  card,  2025.URL https:
//huggingface.co/Kenpath/svara-tts-v1.
## 5