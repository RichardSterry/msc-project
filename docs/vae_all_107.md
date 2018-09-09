# Beta-VAE on All-107

Factor 58 appears to discriminate the gender labels (which the model knew nothing about during training):
<p align="center"><img width="70%" src="vae/all/vae_mu_distribs_by_factor.png" /></p>

### Factor 58 Controls Gender
I picked utterances at random and computed their MAP embeddings (i.e. using mu|x). Then, for each embedding, I made a pair of copies with:
* Factor 58 set to -1
* Factor 58 set to +1 

I then geerated speech samples and compared the results for each pair to hear the impact of varying factor 58.


|  # | Factor 58 = -1 | Factor 58 = +1 |  
| --- | --- | --- | 
| 1 | <audio src=" vae/all/test_latext_58_embedding_612_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_612_1.0.wav" controls></audio> |
| 2 | <audio src=" vae/all/test_latext_58_embedding_1642_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_1642_1.0.wav" controls></audio> |
| 3 | <audio src=" vae/all/test_latext_58_embedding_2566_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_2566_1.0.wav" controls></audio> |
| 4 | <audio src=" vae/all/test_latext_58_embedding_997_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_997_1.0.wav" controls></audio> |
| 5 | <audio src=" vae/all/test_latext_58_embedding_745_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_745_1.0.wav" controls></audio> |
| 6 | <audio src=" vae/all/test_latext_58_embedding_1961_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_1961_1.0.wav" controls></audio> |
| 7 | <audio src=" vae/all/test_latext_58_embedding_89_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_89_1.0.wav" controls></audio> |
| 8 | <audio src=" vae/all/test_latext_58_embedding_2286_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_2286_1.0.wav" controls></audio> |
| 9 | <audio src=" vae/all/test_latext_58_embedding_456_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_456_1.0.wav" controls></audio> |
| 10 | <audio src=" vae/all/test_latext_58_embedding_2203_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_2203_1.0.wav" controls></audio> |


### Factor 36 Has no Impact on Gender

|  # | Factor 36 = -1 | Factor 36 = +1 |  
| --- | --- | --- | 
| 1 | <audio src=" vae/all/test_latext_36_embedding_1973_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_36_embedding_1973_1.0.wav" controls></audio> |
| 2 | <audio src=" vae/all/test_latext_36_embedding_2630_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_36_embedding_2630_1.0.wav" controls></audio> |
| 3 | <audio src=" vae/all/test_latext_36_embedding_2206_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_36_embedding_2206_1.0.wav" controls></audio> |
| 4 | <audio src=" vae/all/test_latext_36_embedding_776_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_36_embedding_776_1.0.wav" controls></audio> |
| 5 | <audio src=" vae/all/test_latext_36_embedding_2389_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_36_embedding_2389_1.0.wav" controls></audio> |