
# Speaker Representations in Neural Text-To-Speech Synthesis
### Richard Sterry, September 2018
Audio samples and supplementary materials in support of my thesis project for the MSc Computational Statistics & Machine Learning, UCL.

My project uses Facebook AI Research's VoiceLoop model as the starting point. Many thanks to FAIR for making their implementation public on GitHub. 
cite paper and link to GitHub

## [VCTK Dataset](vctk.md)
* Samples of <transcript, audio> data
* Speaker metadata

## [Baseline VoiceLoop Models](voiceloop_baseline.md)
* Sample synthesized speech from the two baseline models: VCTK-US-22 and VCTK-All-107
* Training curves?

## [Speaker Representations in VoiceLoop](speakers_in_voiceloop.md)
* Subjective samples to support idea that VoiceLoop can represent different voices
* t-sne, pca plots
* Examples of voice interpolation
* Maybe roll this into the previous section? Or keep separate to match the report?

## [Disentanglement With Labels: Fader Networks](fader_networks.md)
* Samples of speech to show the impact of 'fading' between gender lables

## [Disentanglement Without Labels: BetaVAE](vae_random_sample.md)
* Utterance embeddings: taking speaker labels away
* Basic VAE
* Does BetaVAE help disentangle the generative attributes?
