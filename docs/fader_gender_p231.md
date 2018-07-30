# p231

| id_loop | id_vctk | age | gender | accents | region |
| --- | --- | --- | --- | --- | --- |
| 6 | 231 | 23 | F | English | Southern England |

Examples of synthesized voice data for speaker embeddings at various values of the gender variable.

| id_sentence | transcript |
| --- | --- |
| 108 | Before the game we went for a pint |
| 120 | There were no grounds for his action |
| 233 | I'm so proud to be Scottish tonight |

## Fading Between Genders
Mixing gender composition between pure male and female.

| [M, F] | 108 | 120 | 233 | 
| --- | --- | --- | --- |
| Original | <audio src="audio/fader_networks/p231_108_45_orig.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_orig.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_orig.wav" controls></audio> | 
| **[1, 0]** | <audio src="audio/fader_networks/p231_108_45_gender_concat_1_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_1_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_1_0.wav" controls></audio> |
| [0.8, 0.2] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0.8_0.2.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0.8_0.2.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0.8_0.2.wav" controls></audio> |
| [0.6, 0.4] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0.6_0.4.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0.6_0.4.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0.6_0.4.wav" controls></audio> | 
| [0.5, 0.5] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0.5_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0.5_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0.5_0.5.wav" controls></audio> |
| [0.4, 0.6] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0.4_0.6.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0.4_0.6.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0.4_0.6.wav" controls></audio> | 
| [0.2, 0.8] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0.2_0.8.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0.2_0.8.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0.2_0.8.wav" controls></audio> |
| [0, 1] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0_1.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0_1.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0_1.wav" controls></audio> |

## Extreme Values of Gender Factors
Investigating robustness with extreme values of gender factors. Note there is no gender mixing in these samples.

| [M, F] | 108 | 120 | 233 | 
| --- | --- | --- | --- |
| Original | <audio src="audio/fader_networks/p231_108_45_orig.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_orig.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_orig.wav" controls></audio> | 
| [2, 0] | <audio src="audio/fader_networks/p231_108_45_gender_concat_2_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_2_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_2_0.wav" controls></audio> |
| [1.5, 0] | <audio src="audio/fader_networks/p231_108_45_gender_concat_1.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_1.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_1.5_0.wav" controls></audio> |
| [1, 0] | <audio src="audio/fader_networks/p231_108_45_gender_concat_1.0_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_1.0_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_1.0_0.wav" controls></audio> | 
| [0.5, 0] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0.5_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0.5_0.wav" controls></audio> |
| [0, 0] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0_0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0_0.wav" controls></audio> | 
| [0, 0.5] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0_0.5.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0_0.5.wav" controls></audio> |
| [0, 1] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0_1.0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0_1.0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0_1.0.wav" controls></audio> |
| [0, 1.5] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0_1.5.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0_1.5.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0_1.5.wav" controls></audio> |
| [0, 2.0] | <audio src="audio/fader_networks/p231_108_45_gender_concat_0_2.0.wav" controls></audio> | <audio src="audio/fader_networks/p231_120_45_gender_concat_0_2.0.wav" controls></audio> | <audio src="audio/fader_networks/p231_233_45_gender_concat_0_2.0.wav" controls></audio> |
