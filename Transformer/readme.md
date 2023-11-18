# Attension is all you need (Transformer)

### Origin papaer at : https://arxiv.org/abs/1706.03762

### 0. Abstract

 The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.

>  현재 지배적으로 사용되고있는 시퀀스 변환 모델들은 인코더와 디코더를 포함하는 CNN(합성곱 신경망)구조 이거나, 복잡한 RNN(순환 신경망) 구조 이다.

 The best performing models also connect the encoder and decoder through an attention mechanism.

>  가장 좋은 성능을 내고 있는 모델들 역시 인코더와 디코더를 어텐션 기법을 활용하여 연결 시키고있다.

 We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

> 우리는 recurrence(순환)와 convolution(합성곱)을 완전히 배제하고, 어텐션 기법에만 기초한 Transformer라는 새로운 간단한 신경망 구조를 제안한다. 

 Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

>  두가지 기계학습 과제에 대한 실험은 이 모델이 병렬처리가 더 용이하고, 훈련에 있어 확연하게 적은시간을 필요로 하며, 질적으로 더 뛰어남을 보여준다.

 Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

>  우리의 모델은  WMT 2014 Englishto-German translation task에서 28.4 BLEU점수를 기록하면서 기존 앙상블기법을 포함한 최고의 점수를 2 BLEU만큼 갱신하였다.

 On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

>  WMT 2014 English-to-French translation task에서는, 기록되어있는 최고의 모델들의 훈련비용과 비교하여 매우 적은 비용인, 3.5일간의 8개의 GPU를 활용한 훈련 과정을 거친 우리의 모델이 단일모델로서 41.8 BLEU 점수를 기록하며 SOTA(당시 시점 최고 모델) 모델로 선정되었다.

 We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

>  우리는 Transformer가 다른 과제에도 적합하게 일반화 되는것을 제한된 데이터 및 방대한 양의 데이터를 활용한 영어 구성 구문 분석에 성공적으로 적용하여 증명 하였다.

  

### 1. Introduction

 Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.

>  순환 신경망, 특히 장단기 기억(LSTM) 및 게이트드 순환 신경망(GRU : Gated Recurrent Unit)은 일련의 모델링 및 언어 모델링, 기계 번역과 같은 변환 문제에서 최첨단 방법으로 확립되었다.

 Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.

>  이후로 순환 언어 모델과 인코더 디코더 구조의 한계를 넓히기 위하여 많은 노력이 계속해서 이루어지고 있다.

 Recurrent models typically factor computation along the symbol positions of the input and output sequences.

>  순환 모델은 일반적으로 입력 및 출력 시퀀스에 대한 위치 상징(위치정보)를 따라서 계산이 이루어진다.

 Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$.

>  연산과정에서 시간의 흐름에 따라 위치를 정렬하여, 이전 시퀀스의 은닉 상태 $h_{t-1}$와 현재 위치(시점) $t$ 에서의 입력에 대한 함수를 통하여 숨겨진 상태 $h_t$의 시퀀스를 생성한다.

 This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

>  이러한 연속적인(현재가 과거에 종속적임) 특성은 병렬처리가 불가능하게 하는데, 이는 특히 시퀀스 길이가 길어졌을 경우, 메모리의 제약으로 인하여 예제간 배치처리 가 제한되기에 더욱이 중요하게 작용한다.

Recent work has achieved significant improvements in computational efficiency through factorization tricks and conditional computation, while also improving model performance in case of the latter.

>  최근 연구로 인하여 컴퓨팅 효율은 factorization tricks(인수분해 트릭)과 conditional computation(조건부 연산)을 통하여 획기적인 발전을 이루었으며, 모델 성능 또한 conditional computation를 통하여 향상되었다.

The fundamental constraint of sequential computation, however, remains.

>  그러나 연속적인 특성을 갖는 시퀀스 컴퓨팅의 기본적인 제약은 여전히 남아있다.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences.

>  어텐션 메커니즘은 입력 및 출력 시퀀스에 대한 거리와 상관없이 의존성 모델링이 가능하게 하여, 시퀀스 모델링의 필수 구성 요소가 되었으며, 다양한 과제에 대해서 transduction model이 되었다.  (Transduction : predicting particular examples when specific examples from a domain are given)

In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.

>  그러나 거의 모든경우에 이러한 어텐션 기법은 RNN과 결합되어 사용된다.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

>  이 논문에서 우리는 순환방식을 지양하고 전적으로 어텐션 기법에 의존하여 입출력에 대하여 전역에대한 의존성을 그려내는 Transformer를 소개한다. 

The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

>  Transformer는 확연하게 뛰어난 병렬 처리를 가능하게하며, 8개의 P100 GPU를 이용한 12시간의 훈련을 거쳐 번역의 품질적측면에서 새로운 SOTA에 다가갈수있다.

  

### 2. Background

 The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.

>  시퀀셜 컴퓨팅(시계열적 연산)을 줄이는 것의 목적은 합성곱 신경망을 기본 구성으로 하며 숨겨진 표현을 모든 입출력 위치에 대해 병렬적으로 계산하는 Extended Neural GPU, ByteNet,그리고 ConvS2S의 기초를 구축하는것과 일맥상통 한다.

 In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. 

>  이 모델들의 경우 독립적인 입력과 출력 위치간의 관계를 수립하기 위한 연산이 ConvS2S는 선형적으로 ByteNet은 로그적으로 거리에 비례하여 증가한다. 

 This makes it more difficult to learn dependencies between distant positions. 

> 이러한 특성은 서로 거리가 먼 위치간의 의존성을 학습하기 더욱 힘들게한다.

 In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

>  Transformer에서는 이러한 연산을 일정한 횟수로 감소시키는 반면 attension-weighted(특정 부분에 집중됨) 위치를 평균내어 활용하는 만큼 효과적인 해법이 줄어드는데, 우리는 3.2섹션에 설명된 Multi-Head Attention으로 이 문제를 대응 하였다.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. 

>  Intra-attention이라고도 불리는 Self-attention은 특정 시퀀스를 대변할수있는 요인을 찾기위한 연산으로 단일 시퀀스에 대하여 다른 위치를 관계짓는 Attention 기법이다.

 Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.

>  Self-attention은 지금까지 독해, 추출 요약, 자연어 추론 및 과제와 독립적인 문장 표현과 같이 다양한 과제에서 성공적으로 사용되어왔다.

End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks.

>  End-to-end 메모리 네트워크들은 순차적으로 정렬된 회귀 방식 대신 recurrent attention기법을 기초로 한며 간단한 언어 질의 문제와 거대 모델링 과제에서 좋은 성능을 보였다.

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. 

>  우리가 아는 한, Transformer는 순차적으로 정렬된 RNN이나 합성곱을 사용하지 않고 self-attention기법에 전적으로 의존하여 입출력에 대한 표현을 연산하는 첫 transduction 모델이다.
>
>  *transduction : 데이터와 매칭되는 라벨을 통해서 일반화를 시키는것(induction)이 아닌, 데이터 자체에서 답을 찾는 방법*

In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

>  이후 섹션에서 우리는 Transformer와 self-attention의 동기, 그리고 Transformer가 갖는 이점을 이야기 할 것이다.

  

### 3. Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure.

> 가장 경쟁력 있는 신경망 기반 시퀀스 transduction model들은 인코더-디코더 구조를 갖고있다.

Here, the encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $z = (z_1, ..., z_n)$.

>  인코더는 입력 시퀀스 $(x_1, ..., x_n)$를 연속적인 시퀀스인 $z = (z_1, ..., z_n)$으로 매핑한다.

Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time.

>  디코더는 위의 $z$를 입력받아 출력 시퀀스인 $(y_1, ..., y_m)$의 요소를 하나씩 생성한다.

At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

>  각 단계에서 모델은 자기회귀적이며, 이전 단계에서 생성된 값을 다음 생성을 위하여 입력받는다.

<!-- Figure 1 -->

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

>  Transformer는 위의 그림에 나타나있듯이, 인코더와 디코더 모두 쌓여진(stacked) self-attention과 지점별로 완전이 이어진 레이어를 기본 골자로 한다.

#### 3.1 Encoder and Decoder stacks

**Encoder**: The encoder is composed of a stack of N = 6 identical layers.

> **인코더**: 인코더는 6개(N=6)의 완전히 동일한 레이어가 쌓여져 구정되었다.

Each layer has two sub-layers. 

> 각 레이어는 2개의 서브 레이어를 갖고있다.

The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network.

> 첫번째 서브 레이어는 multi-head self-attention기법을 따르고, 두번쨰 서브 레이어는 간단한 위치에따른 완전히 연결된 순방향 신경망이다.

We employ a residual connection around each of the two sub-layers, followed by layer normalization.

> 우리는 2개의 서브 레이어 각각을 우회하는 residual connection방법을 채용하였으며 이는 layer normalization과 이어진다.
>
> *Residual connection : 기울기 소실을 방지하기위하여 특정 단계의 정보가 일부레이어를 건너뛰고 모델 후반부 레이어에 도달할수 있도록 우회 경로를 제공하는 방법*

That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. 

> 즉, 각 서브레이어의 출력은 다음과 같게 된다. $\hat{y} = LayerNorm(x + Sublayer(x))$. 이때 $Sublayer(x)$는 위 2개의 방식으로 구현된 함수를 의미한다.

 To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model} = 512$.

> residual connection이 가능한 구조를 유지하기위하여 이떄 임베딩 레이어를 포함한 모델 내의 모든 서브 레이어의 출력은 $d_{model} = 512$의 차원을 유지한다.

**Decoder**: The decoder is also composed of a stack of N = 6 identical layers.

> **디코더**: 디코더 역시 인코더와 동일하게 6개(N=6)의 완전히 동일항 레이어로 구성된다.

In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. 

> 디코더에서는 2개의 서브레이어를 갖던 인코더 레이어와는 다르게 인코더 뭉치(stack)의 출력에 대하여 multi-head attention을 수행하는 3번째 서브레이어를 갖는다.

Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.

> 인코더와 비슷하게, 우리는 각 서브레이어를 우회하며 layer normalization으로 이어지는 residual connection을 채용하였다.

We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. 

> 우리는 또한 디코더 스택의 self-attention 서브레이어를 변형(masking 추가)하여 서브 시퀀스의 위치정보에 집중하게 되는 현상을 방지하였다.

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

> Masking은 하나의 위치에 대하여 임베딩 출력이 offset인 점과 더해져, 위치 $i$에 대한 예측이 순전히 $i$위치 이전의 도출된 출력들에만 의존하도록 보장한다.

#### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

> Attension 함수는, query, key, value와 output모두가 벡터일때, 출력(output)으로 query와 key-value 집합을 매핑하는 역할을 한다.

<!-- figure2 -->

The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

> 출력은  가중치가 적용된 value들의 합으로 계산되며, 각 vlaue에 할당된 가중치는 query와 그에 대응하는 key에 대한 Compatiblility function으로 계산된다.
>
> *Compatibility function : q,k를 입력받아 MatMul -> Scale -> Mask -> SoftMax -> 가중치를 생성해내는 부분 (figure2)*

#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention".

> 우리는 우리가 변형한 attention을 "Scaled Dot-Product Attention"으로 부른다.

The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. 

> 입력은 $d_k$차원으로 구성된 query와 key들로 구성되었으며, value들은 $d_v$차원으로 구성되었다.

We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.

> 우리는 query에 대하여 모든 key와의 스칼라곱 연산을 진행하고, 각 연산결과를 $\sqrt{d_k}$로 나누며, 그 결과에 대해 softmax 함수를 적용해서 value들에 대한 가중치를 얻어낸다.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$.

>  본 연구에서, 우리는 attention함수를 query들의 집합에 대해 동시다발적으로 연산을 진행하고, $Q$매트릭스로 모았다.

The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

> key와 value들 또한 매트릭스 $K$와 $V$로 모았으며, 우히는 행렬에 대한 출력을 다음과 같이 계산하였다 : $Attention(Q,K,V) = softmax(frac{QK^T}{\sqrt{d_k}})V$

The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention.

> 가장 일반적으로 자주 사용되는 attention 함수는 additive attention과 dot-product(multiplicative) attention이다.

Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$.

> Dot-product attention은 $\frac{1}{\sqrt{d_k}}$로 조정을 해주는 점만 제외하고, 우리의 알고리즘과 완전이 동일하다.

Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.

> Additive attention은 단일 은닉층과 순방향 신경망을 활용하여 compatibility function을 계산한다.

While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

> 두가지 attention기법은 이론적인 복잡도측면에서 유사하지만, dot-product attention이 고도로 최적화괸 행렬곱 코드를 적용할 수 있기에, 연구에 있어 additive attention보다 빠르고 공간 효율적이다.

While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$.

> 작은 값의 $d_k$에서는 두가지 attention 기법 모두 비슷한 성능을 보이지만, 보정이 없는 큰 값의 $d_k$에서는 additive attention이 dot-product attention의 성능을 뛰어넘는다.

We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

> 우리는 $d_K$값이 클경우, 그에 대한 스칼라곱이 큰촉으로 증가하여 sotmax 함수를 매우 작은 기울기의 영역으로 밀어내는것을 고려하였다.

To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$ .

> 이를 해결하기 위하여 우리는 스칼라 곱을 $\frac{1}{\sqrt{d_k}}$로 보정해 주었다.

#### 3.2.2 Multi-Head Attention

Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively.

> $d_{model}$차원의 key, value와 query들에 대한 단일 attention 함수를 수행하는 대신, 우리는 $d_k$에 대하여 학습된 선형 투영(projections)을 서로다른 $d_k$ 및 $d_v$차원에 대하여 query와 key, value들을 $h$번 선형적으로 투영시키는것이 더 이득이라는 것을 발견했다.

On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values.

> 투영된 버전의 query와 key, value들에 대하여 우리는 병렬적으로 attention 함수를 수행하고, $d_v$차원의 출력 값을 도출한다.

These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

> 이 값들은 이어 붙여지고 다시한번 투영되어 Figure2에 나타나 있듯이 최종 값에 도달한다.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

> Multi-head attention은 모델이 다른 위치에서 다른 투영 공간(시퀀스가 투영된)을 복합적으로 고려할 수 있게 해준다.

With a single attention head, averaging inhibits this.

> 단일 attention head를 평균내는것은 이를 억제 한다.

$$
\begin{align}
MultiHead(Q,K,V) &= Concat(head_1, ..., head_h)W^O\\
where \quad head_1 &= Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})
\end{align}
$$

Where the projections are parameter matrices $W_{i}^{Q} \in \mathbb{R}^{d_{model} \times d_{k}}$, $W_{i}^{K} \in \mathbb{R}^{d_{model} \times d_{k}}$, $W_{i}^{V} \in \mathbb{R}^{d_{model} \times d_{v}}$ and $W^{O} \in \mathbb{R}^{hd_v \times d_{model}}$.

> 투영(projections)는 다음과 같은 파라미터 행렬이다. $W_{i}^{Q} \in \mathbb{R}^{d_{model} \times d_{k}}$, $W_{i}^{K} \in \mathbb{R}^{d_{model} \times d_{k}}$, $W_{i}^{V} \in \mathbb{R}^{d_{model} \times d_{v}}$, $W^{O} \in \mathbb{R}^{hd_v \times d_{model}}$

In this work we employ $h = 8$ parallel attention layers, or heads.

> 우리는 본 연구에서 $h =8$인 병렬 attention 레이어 또는 헤드를 채용했다.

For each of these we use $d_k = d_v = d_{model}/h = 64$.

> 각 요소를 우리는 다음과 같이 설정하였다. $d_k = d_v = d_{model}/h = 64$

Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

> 각 헤드의 감소된 차원으로 인하여 전체적인 연산 비용은 모든 차원에 대한 single-head attention과 비슷해 진다.

#### 3.2.3 Applications of Attention in out Model

The Transformer uses multi-head attention in three different ways:

> Transformer는 multi-head attention을 다음과 같은 3가지 서로다른 방법으로 활용한다.

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.

- > ""인코더-디코더 attention" 레이어에서는 query들이 이전의 디코더 레이어에서 전달되며, key와 value에 대한 메모리는 인코더의 출력에서 전달된다.

  This allows every position in the decoder to attend over all positions in the input sequence.

  > 이는 디코더의 모든 위치에서 입력 시퀀스에 대한 모든 위치를 고려할 수 있게 해준다.

  This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].

  > 이는 [38, 2, 9]와 같은 전형적인 sequence-to sequence모델에서의 인코더-디코더 기법을 모방한 것이다.

- The encoder contains self-attention layers.

- > 인코더는 self-attention 레이어를 포함한다.

  In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.

  > Self-attention 레이어 에서 모든 key, value와 query들은 인코더 내 이전 레이어의 출력인 같은 공간에서 부터 전달된다.

  Each position in the encoder can attend to all positions in the previous layer of the encoder.

  > 인코더 내의 각 위치는 인코더의 이전 레이어의 모든 위치를 참조 할 수 있다.

- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.

- > 이와 유사하게, 디코더 내의 self-attension 레이어는 디코더 내의 각 위치가 해당 위치를 포함한 디코더 내의 모든 위치를 참조 할 수 있게 한다.

  We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.

  > 우리는 자기회귀 요소를 보존하기위하여 디코더 내의 좌측(시계열적 관점에서)으로 흐르는 정보를 방지 해야 했다.

  We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2.

  > 우리는 이를 허가되지 않은 연결에 대해서 softmax 함수로 입력되는 모든 값을 masking out( $-\infty$로 설정)처리하는 것으로 scaled dot-product attentiond에 적용하였다.

#### 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. 

> Attention 서브 레이어와 더불어, 우리의 인코더와 디코더의 각 레이어는 각 위치에 독립적으로 적용되었으며 동일한 구조를 갖는 완전히 연결된 순방향 신경망을 갖고있다.

This consists of two linear transformations with a ReLU activation in between.

> 이는 ReLU 활성함수가 사이에 있는 두개의 선형 변환으로 이루어져 있다.

$$
FFN(x) = max(0,xW_1+b_1)W_2+b_2
$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer.

> 선형 변환의 경우 서로다른 위치에 대하여 모두 동일하지만, 레이어 대 레이어에서는 서로다른 파라미터를 사용한다.

Another way of describing this is as two convolutions with kernel size 1.

> 이는 1의 커널 크기를 갖는 2개의 합성곱이라고도 설명될 수 있다.

The dimensionality of input and output is $d_{model} = 512$, and the inner-layer has dimensionality $d_{f f} = 2048$.

> 입력과 출력은 $d_{model} = 512$의 차원을 갖고 있으며, 내부 레이어의 경우 $d_{ff} = 2048$의 차원을 갖고있다.

#### 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$.

> 다른 시퀀스 transduction model들과 유사하게 우리는 학습된 임베딩을 사용하여 입출력 토큰을 $d_{model}$차원의 벡터로 변환한다.

We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

> 우리는 또한 평범하게 학습된 선형 변환과 더불어 softmax 함수를 활용하여 디코더의 출력을 다음 토큰으로 가능한 토큰의 확률예측값으로 변환한다. 

In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].

> 우리의 모델에서, 우리는 [30]과 유사하게 2개의 임베딩 레이어와 pre-softmax 선형 변환에서 같은 가중치 행렬을 공유한다.

In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.

> 임베딩 레이어에서 우리는 그 가중치들에 $\sqrt{d_{model}}$을 곱연산 한다.

<!-- Table 1-->

#### 3.5 Positional Encoding

