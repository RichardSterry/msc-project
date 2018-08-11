# The VCTK Dataset

* Samples of <transcript, audio> data
* Speaker metadata
 
 ## Examples: <transcript, speaker, audio>
<audio src="vctk/samples/p255_367.wav" controls></audio> 

| id_vctk | age | gender | accents | region | audio | transcript | 
| --- | --- | --- | --- | --- | --- | --- |
| 253 | 22 | F | Welsh | Cardiff | <audio src="vctk/samples/p255_367.wav" controls></audio>  | "She went with him to the store" |

| speaker | transcript | audio | 
| --- | --- | --- | 
| 253 (22/F/Welsh/Cardiff) | "She went with him to the store" | <audio src="vctk/samples/p255_367.wav" controls></audio>  |


![waveform](vctk/samples/wavplot_255_367.png)
![spectrogram](vctk/samples/wavplot_255_367_spectro.png)

## Speakers
![lf0](vctk/vctk_descriptive_age_etc.png)

[VCTK Speaker List](vctk_speaker_metadata_csv.html)
<html>
<iframe style="border-style: none;" src="vctk_speaker_metadata_csv.html" height="400" width="600"></iframe>
</html>

## Transcripts
[VCTK Transcripts](vctk_transcript_csv.html)
<html>
<iframe style="border-style: none;" src="vctk_transcript_csv.html" height="700" width="800"></iframe>
</html>
