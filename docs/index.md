
# Disentangled Speaker Representations in Neural Text-To-Speech Synthesis
### Richard Sterry, September 2018
Audio samples and supplementary materials in support of my project for the MSc Computational Statistics & Machine Learning, UCL.

I used Facebook AI Research's [VoiceLoop](https://github.com/facebookresearch/loop) model as the starting point. Many thanks to FAIR for making their implementation public on GitHub. 


## Part I: Introduction
### [Chapter 1: A Brief History of Speech Synthesis](introduction_resources.md)
* Maybe links to websites/pages mentioned in my intro

### Chapter 2: Maybe link to papers?

### [Chapter 3: My Project](architecture_overview.md)

<hr>

## Part II: Speaker Representations in VoiceLoop 
### Chapter 4: The VoiceLoop Model
#### §4.1 [The VCTK Dataset](vctk.md)
* Samples of <transcript, audio> data
* Speaker metadata

#### §4.2 [WORLD Features](world_features.md)
* Illustration of waveform -> 63 WORLD features
* Example of the intepretability of lf0, the canonical WORLD feature

#### §4.3-4 [Baseline VoiceLoop Models](voiceloop_baseline.md)
* Sample synthesized speech from the two baseline models: VCTK-US-22 and VCTK-All-107
* Training curves?
* Add architecture diagram


### Chapter 5: Speaker Embeddings
#### §5.1 VoiceLoop can Represent Known Speakers
##### [Samples of Baseline Model: VCTK US_21](vctk_us_22_samples.md)
##### [Samples of Baseline Model: VCTK All_107 Samples](vctk_all_107_samples.md)
#### §5.2 [The Speaker Embedding Space is Interpretable](speakers_in_voiceloop.md)
* Subjective samples to support idea that VoiceLoop can represent different voices
* t-sne, pca plots
* Examples of voice interpolation
* Maybe roll this into the previous section? Or keep separate to match the report?

### Chapter 6: VoiceLoop with Utterance Embeddings
#### [Samples of Utterance-US-21](utterance_embeddings_us.md)
#### [Samples of Utterance-All-107](utterance_embeddings_all.md)
* Adapting the VoiceLoop architecuture to embed reference utterance rather than speaker IDs

<hr>

## Part III: Learning Disentangled Speaker Representations 
### Chapter 7:  [Disentanglement Using Labels: Fader Networks](fader_networks.md)
* Samples of speech to show the impact of 'fading' between gender lables

### Chapter 8: [Disentanglement Without Labels: BetaVAE](betavae.md)
* Utterance embeddings: taking speaker labels away
* Basic VAE
* Does BetaVAE help disentangle the generative attributes?
