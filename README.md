# Transformers-BERT-GPT
"Attention is all you need."
Includes code for transformers and BERT model.
Uses pretrained GPT3

For transformers:
Transformers only rely on self-attention and discard convolutional layers. BERT-based transformers have 12 encoders. In one encoder, we will first do self-attention, then simply feed forward and normalize.\\
Self-attentions determines which words should pay attention to. Within same group, $h=h_m$. To calculate attention, we need $q_m$(query), $k_m$(key), and $v_m$(value). These three vectors are created by multiplying embedding with a weight matrix that we will train. In an attention layer, we will first update $h_m$ by adding $\Delta h_m$, which can be calculated as:$\sum_{t=1}^{M} \frac{exp(<q_m,k_t>)}{\sum_{t'=1}^{M}exp(<q_m,k_{t'}>)}v_t$. Then we normalize the layer and let each $h_m$ think for itself: $h_m=hm+F(h_m)$. In the end, we normalize the layer again to conclude an attention layer.\\
In some cases, we may need multi-heads, that is, some $h_m$ may have more than one set of query, key and values. We denote them as $q_{mj}$, $k_{mj}$, and $v_{mj}$, respectively. Multi-heads attention is useful because sometimes one word can have multiple meanings in real life. In this situation, $\Delta h_m$ is defined as:$\sum_{j=1}^{M} \sum_{t=1}^{M} \frac{exp(<q_{mj},k_{tj}>)}{\sum_{t'=1}^{M}exp(<q_{mj},k_{t'j}>)}v_{tj}$\\
After calculating the attention, we feed forward and normalize the result. This concludes one encoder, then we stack the encoders together.
