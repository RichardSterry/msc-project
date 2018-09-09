# Beta-VAE on All-107

![0.5](vae/all/vae_mu_distribs_by_factor.png)

### Factor 58 Controls Gender
I picked utterances at random, computer their MAP embeddings (i.e. using mu|x) and generated samples using these embeddings with two changes:
* Factor 58 set to -1
* Factor 58 set to +1 

|  # | Factor 58 = -1 | Factor 58 = +1 |  
| --- | --- | --- | 
| 1 | <audio src=" vae/all/test_latext_58_embedding_612_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_612_+1.0.wav" controls></audio> |
| 2 | <audio src=" vae/all/test_latext_58_embedding_1642_-1.0.wav" controls></audio> | <audio src=" vae/all/test_latext_58_embedding_1642_+1.0.wav" controls></audio> |
