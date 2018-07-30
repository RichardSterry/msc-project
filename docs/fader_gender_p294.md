# p294

| id_loop | id_vctk | age | gender | accents | region |
| --- | --- | --- | --- | --- | --- |
| 63 | 294 | 33 | F | American | San Francisco |

Examples of synthesized voice data for speaker embeddings at various values of the gender variable.

| id_sentence | transcript |
| --- | --- |
| 64 | It is all about having physical evidence |
| 86 | We have to focus on what is important |
| 372 | But opponents were quick to criticize |

## Fading Between Genders
Mixing gender composition between pure male and female.

| [M, F] | 64 | 86 | 372 | 
| --- | --- | --- | --- |
| Original | <audio src="audio/fader_networks/p294_064_63_orig.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_orig.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_orig.wav" controls></audio> | 
| **[1, 0]** | <audio src="audio/fader_networks/p294_064_63_gender_concat_1_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_1_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_1_0.wav" controls></audio> |
| [0.8, 0.2] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0.8_0.2.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0.8_0.2.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0.8_0.2.wav" controls></audio> |
| [0.6, 0.4] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0.6_0.4.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0.6_0.4.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0.6_0.4.wav" controls></audio> | 
| [0.5, 0.5] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0.5_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0.5_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0.5_0.5.wav" controls></audio> |
| [0.4, 0.6] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0.4_0.6.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0.4_0.6.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0.4_0.6.wav" controls></audio> | 
| [0.2, 0.8] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0.2_0.8.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0.2_0.8.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0.2_0.8.wav" controls></audio> |
| [0, 1] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0_1.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0_1.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0_1.wav" controls></audio> |

## Extreme Values of Gender Factors
Investigating robustness with extreme values of gender factors. Note there is no gender mixing in these samples.

| [M, F] | 64 | 86 | 372 | 
| --- | --- | --- | --- |
| Original | <audio src="audio/fader_networks/p294_064_63_orig.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_orig.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_orig.wav" controls></audio> | 
| [2, 0] | <audio src="audio/fader_networks/p294_064_63_gender_concat_2_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_2_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_2_0.wav" controls></audio> |
| [1.5, 0] | <audio src="audio/fader_networks/p294_064_63_gender_concat_1.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_1.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_1.5_0.wav" controls></audio> |
| [1, 0] | <audio src="audio/fader_networks/p294_064_63_gender_concat_1.0_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_1.0_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_1.0_0.wav" controls></audio> | 
| [0.5, 0] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0.5_0.wav" controls></audio> |
| [0, 0] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0_0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0_0.wav" controls></audio> | 
| [0, 0.5] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0_0.5.wav" controls></audio> |
| [0, 1] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0_1.0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0_1.0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0_1.0.wav" controls></audio> |
| [0, 1.5] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0_1.5.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0_1.5.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0_1.5.wav" controls></audio> |
| [0, 2.0] | <audio src="audio/fader_networks/p294_064_63_gender_concat_0_2.0.wav" controls></audio> | <audio src="audio/fader_networks/p294_086_63_gender_concat_0_2.0.wav" controls></audio> | <audio src="audio/fader_networks/p294_372_63_gender_concat_0_2.0.wav" controls></audio> |
