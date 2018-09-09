
# Disentangled Speaker Representations in Neural Text-To-Speech Synthesis
### Richard Sterry, September 2018
Audio samples and supplementary materials in support of my project for the MSc Computational Statistics & Machine Learning, UCL.

I used Facebook AI Research's [VoiceLoop](https://github.com/facebookresearch/loop) model as the starting point. Many thanks to FAIR for making their implementation public on GitHub. 


## Part I: Introduction
### [Chapter 1: A Brief History of Speech Synthesis](introduction_resources.md)

### Chapter 2: Disentanglement

### [Chapter 3: My Project](architecture_overview.md)

<hr>

## Part II: Speaker Representations in VoiceLoop 
### Chapter 4: The VoiceLoop Model
#### §4.1 [The VCTK Dataset](vctk.md)
#### §4.2 [WORLD Features](world_features.md)
#### §4.3-4 [Baseline VoiceLoop Models](voiceloop_baseline.md)


### Chapter 5: Speaker Embeddings
#### §5.1 VoiceLoop can Represent Known Speakers
* ##### [Samples of Baseline Model: VCTK US_21](vctk_us_22_samples.md)
* ##### [Samples of Baseline Model: VCTK All_107 Samples](vctk_all_107_samples.md)

#### §5.2 [The Speaker Embedding Space is Interpretable](speakers_in_voiceloop.md)

### Chapter 6: [VoiceLoop with Utterance Embeddings](utterance_embeddings.md)

<hr>

## Part III: Learning Disentangled Speaker Representations 
### Chapter 7:  [Disentanglement Using Labels: Fader Networks](fader_networks.md)

### Chapter 8: [Disentanglement Without Labels: BetaVAE](betavae.md)
