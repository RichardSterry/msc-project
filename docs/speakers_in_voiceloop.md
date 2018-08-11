# Speaker Representations in VoiceLoop

* Subjective samples to support idea that VoiceLoop can represent different voices
* t-sne, pca plots
* Examples of voice interpolation
* Maybe roll this into the previous section? Or keep separate to match the report?


## Speaker Embeddings
![wavform](ppt/baseline_speakers_pca_gender.png)
![wavform](ppt/baseline_speakers_pca_accent.png)
![wavform](ppt/baseline_speakers_pca_accent_male.png)
![wavform](ppt/baseline_speakers_pca_accent_female.png)
![wavform](ppt/baseline_cosine_similarity.png)

## Gender Transformation
![wavform](ppt/baseline_speakers_pca_gender_transformation.png)

Speaker 39: Irish Female
"How trying to stay cool could make the world even hotter."
Speaker 39<audio src="ppt/baseline_gender_tranform_39_base.wav" controls></audio>
+female->male transformation<audio src="ppt/baseline_gender_tranform_39.wav" controls></audio>
+2x female->male transformation<audio src="ppt/baseline_gender_tranform_39_x2.wav" controls></audio>

# Accent Transformation
Speaker 34: English Female

"More than two million player ratings were awarded by users."
Speaker 34<audio src="ppt/baseline_accent_transform_34_base.wav" controls></audio>
+English->American transformation<audio src="ppt/baseline_accent_transform_34.wav" controls></audio>

