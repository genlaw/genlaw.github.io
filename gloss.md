--- 
title: "GenLaw: Glossary" 
tags: [machine learning, datasets, generative ai, artificial intelligence, copyright, privacy, law] 
return-genlaw: true 
return-footer: true
--- 
<p style="text-align:right">Also see the [resources](resources.html).</p>

We roughly divide our glossary into sections: 

* [Machine Learning and Generative AI](#gloss:machine-learning) 
* [Open versus Closed](#app:os) 
* [Legal Concepts in Intellectual Property and Software](#gloss:legal-concepts) 
* [Privacy](#app:privacy) 
* [Metaphors](#app:metaphors)

## Machine Learning and Generative AI {#gloss:machine-learning}

### Algorithm {#algorithm}

An **algorithm** is a formal, step-by-step specification of a process. Machine learning uses algorithms for [training](#training) [models](#model) and for applying models (a process called [inference](#inference)). In training, the algorithm takes a model [architecture](#architecture), [training data](#datasets), [hyperparameters](#hyperparameter), and a random seed (to enable random choices during statistical computations) to produce trained model [parameters](#parameters).

In public discourse around social media, the term algorithm is often used to refer to methods for optimizing the probability that a user will engage with a post; however, it is important to note that algorithms describe many processes including, for example, the process of sorting social media posts by date.

### Alignment {#alignment}

**Alignment** refers to the process of taking a [pre-trained](#pt-ft) [model](#model) and further tuning it so that its outputs are *aligned* with a policy set forth by the model developer. Alignment can also refer to the *state of being aligned* --- some academic papers might compare and contrast between an aligned and an unaligned model. The goals included in an alignment policy (sometimes called a constitution) vary from developer to developer, but common ones include:

-   Following the intent of user-provided instructions; 
-   Abiding by human values (e.g., not emitting swear-words); 
-   Being polite, factual, or helpful; 
-   Avoiding generating copyrighted text.

While the goals of alignment are vast and pluralistic, current techniques for achieving better alignment are broadly applicable across goals. These techniques include [reinforcement learning](#rl) with human feedback and full model [fine-tuning](#pt-ft) Specifying the desired properties of alignment often requires a special dataset. These datasets may include user-provided feedback, supplied through the user interface in a generative-AI product.

### Application Programming Interface (API) {#api}

Companies choose between releasing generative-AI [model](#model) functionality in a variety of ways: They can release the model directly by [open-sourcing](#open-model) it; they can embed the model in a software system, which they release as a product; or, they can make the model (or the system its embedded in) available via an **Application Programming Interface (API)**. When a model is open-source, anyone can take the [checkpoint](#checkpoint) and load it onto a personal computer in order to use it for [generation](#Generations). In contrast, when a company only makes their generative-AI model or system available via an API, that means that users access it in code. The user writes a query in a format specified by the company, then sends the query to the company's server. The company then runs the model or system on their own computers, and provides a response to the user with the generated content. API access usually requires accepting the company's [Terms of Service](#tos), and companies may add extra layers of security on top of the model (such as rejecting queries identified as being in violation of the ToS).

### Architecture {#architecture}

The design of a [model](#model) is called its **architecture**. For a [neural network](#nn), architectural decisions include the format of inputs that can be accepted (e.g., images with a certain number of pixels), the number of layers, how many [parameters](#parameters) per layer, how the parameters in each layer are connected to each other, and how we represent the intermediate state of the model as each layer transforms input to output. The most common architecture for language tasks is called a Transformer [@vaswani2017attention], for which there are many variations.

Many contemporary models appear in **model families** that have similar architectures but different sizes, often differentiated by the total number of parameters in the model. For example, Meta originally released four sizes of the LLaMA family that had almost the exact same architectures, differing only in the number of layers and size of the intermediate [vector representations](#vector). More layers and wider internal representations can improve the capability of a model, but can also increase the amount of time it takes to [train](#training) the model or to do [inference](#inference).

### Attention {#attention}

An **attention** mechanism is a sub-component of a [transformer](#transformer) architecture. This mechanism allows a [neural network](#nn) to selectively focus on (i.e., **attend** to) specific tokens in the input sequence by assigning different attention weights to each token. Attention also refers to the influence an input token has on the token that is generated by the model.

### Base Model {#base-model}

A **base model** (sometimes called a "[foundation](#foundation)" or "pre-trained model") is a [neural network](#nn) that has been [pre-trained](#pt-ft) on a large general-purpose dataset, such as a [web crawl](#web-crawl). Base models can be thought of as the first step of training and a good building block for other models. Thus, base models are not typically exposed to Generative AI users; instead they are adapted to be more usable either through [alignment](#alignment) or [fine-tuning](#pt-ft) or performing [in-context learning](#in-context) for *specific* tasks. For example, OpenAI trained a base model called GPT-3 then adapted it to follow natural-language instructions from users to create a subsequent model called InstructGPT [@instructgpt].

See also [pre-training and fine-tuning](#pt-ft).

### Checkpoint {#checkpoint}

While a [model](#model) is being trained, all of its [parameters](#parameters) are stored in the computer's memory, which gets reset if the program terminates or the computer turns off. To keep a model around long-term, it is written to long-term memory, i.e., a hard drive in a file called a **checkpoint.** Often, during training, checkpoints are written to disk every several thousand steps of training. The minimum bar for a model to be considered [open-source](#open-model) is if there has been a public release of one of its checkpoints, as well as the code needed to load the checkpoint back into memory.

### Context Window {#context}

Also called **prompt length**. Generative AI [prompts](#prompt) typically have a fixed **context window**. This is the maximum accepted input length for the model and arises because models are trained with data examples that are no longer than this maximum context window. Inputs longer than this maximum context window may result in generations with performance degradations.

### Data Curation and Pre-Processing {#data-curation}

**Data curation** is the process of creating and curating a dataset for training a model. In the past, when datasets were smaller, they could be manually curated, with human annotators assessing the quality of each example. Today, the final datasets used to train generative machine learning models are typically automatically curated. For example, data examples identified as "toxic" or "low quality" may be removed. This filtering is done using an [algorithm](#algorithm). These algorithms may include heuristic rules (e.g., labeling examples containing forbidden words as toxic) or may use a different machine learning model trained to classify training examples as low-quality or toxic. Another common curation step is to deduplicate the dataset by identifying examples that are very similar to each other and removing all but one copy [@lee2022dedup].

**Data pre-processing** involves transforming each example in the dataset to a format that is useful for the task we want our machine learning model to be able to do. For example, a web page downloaded from a [web crawl](#web-crawl) might contain HTML markup and site navigation headers. We may want to remove these components and keep only the content text.

Data curation and pre-processing happens directly to the data and are independent of any model. Different models can be trained on the same dataset. For more information about dataset curation and pre-processing, see @lee2023explainers [Chapter 1]

### Datasets {#datasets}

**Datasets** are collections of data [examples](#examples). Datasets are used to [train](#training) machine-learning models. This means that the [parameters](#parameters) of a machine learning model depend on the dataset. Different machine-learning models may require different types of data and thus different datasets. The choice of dataset depends on the task we want the machine learning model to be able to accomplish. For example, to train a model that can generate images from natural-language descriptions, we would need a dataset consisting of aligned image-text pairs.

Datasets are often copied and reused for multiple projects because (a) they are expensive and time-consuming to create, and (b) reuse makes it easier to compare new models and algorithms to existing methods (a process called **benchmarking**). Datasets are usually divided into [training](#training) and **testing** portions. The testing portion is not used during training, and is instead used to measure how well the resulting model [generalizes](#generalization) to new data (how well the model performs on examples that look similar to the training data but were not actually used for training).

Datasets can be either created directly from a raw source, such as Wikipedia or a [web crawl](#web-crawl), or they can be created by assembling together pre-existing datasets. Here are some popular "source" datasets used to train generative machine learning models:

-   [*Wikipedia*](https://huggingface.co/datasets/wikipedia) 
-   [*Project Gutenberg*](https://arxiv.org/abs/1812.08092v1) 
-   [*Common Crawl*](https://commoncrawl.org/) 
-   [*C4*](https://github.com/allenai/allennlp/discussions/5056) 
-   [*ImageNet*](https://www.image-net.org/) 
-   [*LAION-400M*](https://laion.ai/blog/laion-400-open-dataset/) 
-   [*ROOTS*](https://huggingface.co/papers/2303.03915)

Here are some popular datasets that were created by collecting together several pre-existing datasets: 

- The Pile (The Pile was formerly listed online, but was removed following the Huckabee v. Meta Platforms, Inc., Bloomberg L.P., Bloomberg Finance, L.P., Microsoft Corporation, and The EleutherAI     Institute class action complaint [@huckabee]. Notably, a     de-duplicated version of the dataset is still available on     HuggingFace via EleutherAI, as of the writing of this     report [@deduppile].) 
- [*Dolma*](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64) 
- [*RedPajama*](https://github.com/togethercomputer/RedPajama-Data)

Collection-based datasets tend to have a separate license for each constituent dataset rather than a single overarching license.

### Decoding {#decoding}

A **decoding algorithm** is used by a [language model](#lm) to generate the next word given the previous words in a prompt. There are many different types of decoding algorithms including: greedy algorithm, beam-search, and top-k [@decoding]. For language models, the term **decoding** may be used interchangeably with [inference](#inference), [generation](#Generations), and [sampling](#sampling).

### Diffusion-Based Modeling {#diffusion}

**Diffusion-based modeling** is an algorithmic process for model training. Diffusion is *not* itself a [model architecture](#architecture), but describes a process for training a model architecture (typically, an underlying [neural network](#nn)) [@sohldickstein2015dpm; @rombach2022diffusion; @ho2020denoising; @song2019grads]. Diffusion-based models are commonly used for image generation, such as in the Stable Diffusion text-to-image models [@stable-diffusion].

### Embedding {#embedding}

(Also called **vector representation**.) Embeddings are numerical representations of data. There are different types of embeddings such as word embeddings, sentence embeddings, image embeddings, etc. [@murphy2022mlintro p. 26, p. 703-10]. Embeddings created with machine learning models seek to model statistical relationships between the data seen during training.

"Common embedding strategies capture semantic similarity, where vectors with similar numerical representations (as measured by a chosen distance metric) reflect words with similar meanings.

Needless to say, such quantified data are not identical to the entities they reflect, *however*, they can capture certain useful information about said entities" Quoted from @lee2023talkin [p. 7]:

### Examples {#examples}

An example is a self-contained piece of data. Examples are assembled into [datasets](#datasets). Depending on the dataset, an example can be an image, a piece of text (such as content of a web page), a sound snippet, a video, or some combination of these. Often times examples are **labeled**---they have an input component that can be passed into a machine learning model, and they have a target output, which is what we would like the model to predict when it sees the input. This format is usually referred to as an "input-output" or "input-target" pair. For example, an input-target example pair for training an image classification model would consist of an image as the input, and a label (e.g. whether this animal is a dog or a cat) as the target.

### Generalization {#generalization}

Generalization in machine learning refers to a model's ability to perform well on unseen data, i.e. data it was not exposed to during training. **Generalization error** is usually measured evaluating the model on *training* data and comparing it with the evaluation of the model on *test* data [@generalization]. Generalization is

### Generation {#generation}

**Generative models** produce complex, human interpretable outputs such as full sentences or natural-looking images, called **generations.** Generation also refers to the process of applying the generative-AI model to an input and generating an output. The input to a generative model is often called a [prompt](#prompt). More traditional, machine learning models are limited to ranges of numeric outputs (**regression**) or discrete output labels like "cat" and "dog" (**classification**). More commonly the word [inference](#inference) is used to describe applying a traditional machine learning model to inputs.

Generative models could output many different generations for the same prompt prompt that may all be valid to a user. For example, there may be many different kinds of cats that would all look great wearing a hat. Thus, evaluating the performance of generative models can be challenging. Recent developments in generative AI have made these outputs look much better. The process of producing generations is much more difficult and even high-quality generations can reach an **uncanny valley** with subtly wrong details like seven-fingered hands.

See also: [inference](#inference).

### Fine-Tuning {#ft}

See description alongside [pre-training](#pt-ft).

### Foundation Model {#foundation}

**Foundation model** is a term coined by researchers at Stanford University [@bommasani2021foundation] to refer to neural networks that are trained on very large general-purpose datasets, after which they can be adapted to many different applications. Another commonly used word is "[base model](#base-model)."

### Hallucination {#hallucination}

There are two definitions of the word **hallucination** [@hallucination]. First, a generation that does not accord with our understanding of reality may be termed a hallucination. For example, an image of a traffic light not connected to anything in response to the prompt: "a sunny street corner in San Francisco," or "bibliographies" consisting of research papers that do not exist. These hallucinations may occur because generative models do not have explicit representations of facts or knowledge. Second, generations that have nothing to do with the input may also be termed a hallucination. For example, an image generated of a cat wearing a hat in response to the prompt "Science fiction from the 80s" may be termed a hallucination.

### Hyperparameter {#hyperparameter}

Neural networks contain both [parameters](#parameters) (or weights) and **hyperparameters.** **Parameters** are the numbers in a network network whose values are updated over and over again during training. Hyperparameters are settings for the model or training process that are manually specify these prior to training. These settings are often written as numbers (such as the number of layers in the neural network), but these are numbers that are *not* learned. Examples of hyperparameters for the model include: properties of the architecture, such as input sequence length, the number of model parameters per **layer**, and number of layers. Examples of hyperparameters that determine the behavior of the training algorithm include: the choice of optimization algorithm and the learning rate, which controls how much we update model parameters after each input/output training example. The process of picking hyperparameters is typically called **hyperparameter optimization**, which is its own field of research.

### In-Context Learning (Zero-Shot / Few-shot) {#in-context}

A [base model](#base-model) can be used directly without creating new [checkpoint](#checkpoint) through [fine-tuning](#pt-ft). **In-context learning** is a method of adapting the model to a specific application by providing additional [context window](#context)to the model through the [prompt](#prompt). This can be used instead of the more computationally expensive [fine-tuning](#pt-ft) process, though it may not be as effective. In-context learning involves creating an input or [prompt](#prompt) to a model in a way that constrains a desired output. Typically the input includes either **instructions** (zero-shot: a natural language description of the output) or a small number of **examples** of input-output pairs (few-shot). "**Shot**" refers to the number of examples provided.

### Inference {#inference}

More traditional, machine learning (like **regression** and **classification**) may have used the word **inference** instead of [generation](#Generations) to refer to the process of applying a trained model to input data. However, both **inference** and **generation** are used to describe the generation process for a generative-AI model.

See also: [generation](#Generations).

### Language Model {#lm}

A language model (LM) is a type of [model](#model) that takes a sequence of text as input and returns a prediction for what the next word in the text sequence should be. This prediction is usually in the form of a probability distribution. For example, when passed the input "It is raining," the language model might output that the probability of "outside" is 70%, and the probability of "inside" is 5%. Language models used to be entirely statistical; the probability of a "outside" coming next in the phrase would be computed by counting the number of times "outside" occurred after the sequence "It is raining" in their training [dataset](#datasets). Modern language models are implemented using neural networks, which have the key advantage that they can base their output probabilities on complex relationships between sequences in the training data, rather than just counting how often each possible sequence occurs. As an illustrative example, a neural network may be able to use information from a phrase like "It's a monsoon outside" occurring in the training data to increase the probability of the word "outside."

A language model can be used for generation by employing an [algorithm](#algorithm) that selects a word to generate given the probabilities outputted by the model for some prompt sequence. After each word is selected, the algorithm appends that word to the previous prompt to create a new prompt, which is then used to pick the next word, and so on. Such an algorithm is referred to as a [decoding algorithm](#decoding).

Language model generation can be used to implement many tasks, including autocomplete (given the start of a sentence, how should the sentence be completed?) and translation (translate a sentence from English to Chinese). The probabilities outputted by the language model can also be used directly for tasks. For example, for a sentiment classification task, we might ask the language model whether "but" or "because" is a more probable next word given the prompt "the food was absolutely delicious."

For more information, see @riedl2023transformers.

### Large Language Model (LLM) {#llm}

This term has become popular as a way to distinguish older language models from more modern language models, which use the [transformer](#transformer) architecture with many parameters and are trained on web-scale datasets. Older models may have used different [model architectures](#architecture), or may have used fewer [parameters](#parameters) and were often trained on smaller, more narrowly scoped datasets. Consensus for what constitutes "large" has shifted over time as previous generations of large language models are replaced with models with even more parameters and trained for even more steps.

### Loss {#loss}

Neural networks take an input and predict an output. The distance between this output and the output we *expect* the model to predict (i.e., the target) is called the loss. Neural networks are [trained](#pt-ft) by an algorithm which repeatedly passes [examples](#examples) into the the network, measures the loss compared to the expected output, and then updates the network's [weights](#parameters) so as to reduce the size of the loss on these examples. The goal of this training process is to minimizing the loss over all the exampling in the training dataset.

Some research areas (for example, reinforcement learning) refer to "maximizing a reward" rather than "minimizing a loss." These concepts are largely interchangeable; a loss can be turned into a reward by adding a negative sign to it.

### Memorization {#memorization}

A training [example](#examples) may be **memorized** by a model if information about that training example can be **discovered** inside the model through any means. A training example is said to be **extracted** from a model if the generative model can be prompted to generate an output that looks exactly or almost exactly the same as the training example. A training example may be **regurgitated** by the model if the generation looks very similar or almost exactly the same as the training example (with or without the user's intention to extract that training example from the model). To use all these words together, a training example is *memorized* by a model and can be *regurgitated* in the generation process regardless of whether the intent is to *extract* the memorized example.

The word memorization itself may be used to refer to other concepts that we may colloquially understand as "memorization." For example, facts and style (artists style) may also be memorized, regurgitated, and extracted.

### Model {#model}

The **model** is at the core of contemporary machine learning. A model is a mathematical tool that takes an **input** and produces an **output**. A simple example might be a model that tells you whether a temperature is above or below average for a specific geographic location. In this case the input is a number (temperature) and the output is binary (above/below).

There could be many versions of this model depending on geographic location. The behavior of the model is defined by an internal [parameter](#parameter) (or, [weight](#weights)). In our temperature example, the model has one parameter, the average temperature for the location. The process for setting the value of the parameter for a specific version of the model is called [training](#training). In this case we might train a model for New York City by gathering historical temperature data and calculating the average. The process of gathering historical data is called [data collection](#data-curation). The process of training --- in this case, calculating the average --- is an [algorithm](#algorithm).

A saved copy of a model's trained parameters is called a [checkpoint](#checkpoint). We might save separate checkpoints for different cities or save new checkpoints for our New York City model if we retrain with new data. The process of applying the model to new inputs is called [inference](#inference). To create an output for a temperature, we also apply an algorithm: subtract the parameter from the input, and return "above" if the difference is positive.

Our temperature example is a very simple model. In machine learning, models can be arbitrarily complex. A common type of model [architecture](#architecture) is a [neural network](#nn), which (today) can have billions of parameters.

It is important to note that models are often embedded within **software systems** that can get deployed to public-facing users. For example, GPT-4 is a generative-AI model that is embedded within the ChatGPT system, which also has a user interface, developer [APIs](#api), and other functionality, like **input and output filters**. The other components are not part of the model itself, but can work in concert with the model to provide overall functionality.

### Multimodal {#multimodal}

Generative-AI models may generate content in one modality (text, images, audio, etc.) or in multiple modalities. For example, DALL-E is a multimodal model that transforms text to images.

### Neural Network {#nn}

A neural network is a type of [model architecture](#architecture). Neural networks consist of **layers**, where the output of one layer is used as the input of the next layer. Each layer consists of a set of classifiers (**neurons**) that each performs a simple operation independently of one another. A neural-network model synthesizes multiple simple decisions by passing the input through a series of intermediate transformations. The outputs of all classifiers at layer *n* are then passed to each classifier in layer *n+1*, and so forth. Each classifier in each layer has [parameters](#parameters) that define how it responds to input.

### Parameters {#parameters}

**Parameters** are numbers that define the specific behavior of a model. For example, in the linear equation model $y=mx+b$, there are two parameters: the slope $m$ and the **bias** (or, **intercept**) $b$. A more complex example might be a model that predicts the probability a person makes a bicycle trip given the current temperature and rainfall. This could have two parameters: one representing the effect of temperature and one representing the effect of rainfall. Contemporary [neural network](#nn) models have millions to billions of parameters. Model parameters are often interchangeably referred to as model [weights](#weights). The values of parameters are saved in files called [checkpoint](#checkpoint).

For more on the distinction between **parameters** and **hyperparameters**, see [hyperparameters](hyperparameter).

### Pre-Processing {#preprocessing}

See [data curation](#data-curation).

### Pre-Training and Fine-Tuning {#pt-ft}

Current protocols divide [training](#training) into a common **pre-training** phase that results in a general-purpose or [base model](#base-model) (sometimes called a foundation or pre-trained model) and an application-specific **fine-tuning** phase that adapts a pre-trained model [checkpoint](#checkpoint) to perform a desired task using additional data. This paradigm has become common over the last five years, especially as model architectures have become larger and larger. This is because, relative to pre-training datasets, fine-tuning datasets are smaller making fine-tuning faster and less expensive: It is much cheaper to fine-tune an existing base model for a particular task than it is to train a new model from scratch.

As a concrete example, fine-tuning models to "follow instructions" has become an important special case with the popularity of ChatGPT (this is called **instruction tuning**). Examples of interactions in which someone makes a request and someone else follows those instructions are relatively rare on the Internet compared to, for example, question / answer forums. As a result, such data sets are often constructed specifically for the purpose of language model fine-tuning, and may provide substantial practical benefits for commercial companies.

Because pre-trained models are most useful if they provide a good basis for many distinct applications, model builders have a strong incentive to collect as much pre-training data from as many distinct sources as possible. Fine-tuning results in a completely new model checkpoint (potentially gigabytes of data that must be loaded and served separately from the original model), and requires hundreds to thousands of application-specific examples.

However, the distinction between pre-training and fine-tuning is not well defined. Models are often trained with many (more than two) training stages. For example, the choice to call the first two of, say three, training stages pre-training and the last stage fine-tuning is simply a choice. Finally, pre-training should not be confused with [data curation or pre-processing](#data-curation).

### Prompt {#prompt}

Most generative-AI systems take as input some text, which is then used to condition the output. This input text is called the **prompt**.

### Reinforcement Learning {#rl}

Reinforcement learning (RL) is a method for incorporating feedback into systems. For generative models, RL is commonly used to incorporate **human feedback** (HF) about whether the generations were "good" or "useful" to improve future generations. For example, ChatGPT collects "thumbs-up" and "thumbs-down" feedback on interactions in the system. User feedback is just one way of collecting human feedback, model creators can also pay testers to rate generations as well.

### Regurgitation {#regurgitation}

See [memorization](#memorization).

### Sampling {#sampling}

### Scale {#scale}

Machine-learning practitioners use the term \"scale\" to refer to the [number of parameters](#parameters) in their model, the size of their training data (commonly measured in terms of number of [examples](#examples) or storage size on disk), or the computational requirements to train the model. The scale of the model's parameter count and the training dataset size directly influence the computational requirements of training. The scale of computation needed for training can be measured in terms of the number of GPU-hours (on a given GPU type), the number of computers/GPUs involved in total, or the number of FLOPs (floating point operations).

Machine-learning practitioners will sometimes talk about "scaling up" a model. This usually means figuring out a way to increase one of the properties listed above. It can also mean figuring out how to increase the fidelity of training examples--e.g. training on longer text sequences or on higher-resolution images.

### Supply Chain {#supply-chain}

Generative-AI systems are created, deployed, and used in a variety of different ways. Other work has written about how it is useful to think of these systems in terms of a **supply-chain** that involves many different stages and actors. For more information about the generative-AI supply chain, please see @lee2023talkin.

### Tokenization {#tokens}

For language models, a common pre-processing step is to break documents into segments called **tokens**. For example, the input "I like ice cream." might be tokenized into \["I", "like", "ice", "cream", "."\]. The tokens can then be mapped to entries in a **vocabulary**. Each entry, or token, in the vocabulary is given an ID (a number representing that token). Each token in the vocabulary has a corresponding [embedding](#embedding) that is a learned [parameter](#parameters). The embedding turns words into numeric representations that can be interpreted and modified by a model.

Each model family tends to share a vocabulary, which is optimized to represent a particular training corpus. Most current models use **subword tokenization** to handle words that would otherwise not be recognized. Therefore, a rare or misspelled word might be represented by multiple tokens, for example \["seren", "dipity"\] for "serendipity." The number of tokens used to represent an input is important because it determines how large the effective [context window](#context) of a model is.

Tokens are also used for other modalities, like music. Music tokens may be **semantic tokens** that may be created using yet another neural network.

### Transformer {#transformer}

A **transformer** is a popular [model architecture](#architecture) for image, text, and music applications, and the transformer architecture underlies models like ChatGPT, Bard, and MusicLM. An input (text or image) is broken into segments (word [tokens](#tokens) or image patches) as a pre-processing step. These input segments are then passed through a series of layers that generate [vector representations](#embeddings) of the segments. The model has trainable parameters that determine how much [attention](#attention) is paid to parts of the input. Like many other generative-AI models, the transformer model is trained with a loss function that rewards reproducing a target training example.

### Training {#training}

Machine-learning models all contain [parameters](#parameters). These parameters are initialized to random numbers when the network is first created. During a process called training, these parameters are repeatedly updated based on the training data that the model has seen. Each update is designed to increase the chance that when a model is provided some input, it outputs a value close to the target value we would like it to output. By presenting the model with all of the examples in a [dataset](#dataset) and updating the parameters after each presentation, the model can become quite good at doing the task we want it to do.

A common [algorithm](#algorithm) for training neural network models is **stochastic gradient descent,** or SGD. Training data sets are often too large to process all at once, so SGD operates on small **batches** of dozens to hundreds of **examples** at a time. Upon seeing a training example, the algorithm generates the model's output based on the current setting of the parameters and compares that output to the desired output from the training data (In the case of a language model, we might ask: did we correctly choose the next word?). If the output did not match, the algorithm works backwards through the model's layers, modifying the model's parameters so that the correct output becomes more likely. This process can be thought of as leaving "echoes" of the training examples encoded in the parameters of the model.

### Vector Representation {#vector}

See [embedding](#embedding).

### Web Crawl {#web-crawl}

A web crawl is a catalog of the web pages accessible on the internet. It is created by a web crawler, an [algorithm](#algorithm) that systematically browses the Internet, trying to reach every single web page. For example, Google Search functions using a web crawl of the internet that can be efficiently queried and ranked. Web crawls are very expensive and compute-intensive to create, and so most companies keep their crawls private. However, there is one public web crawl, called [*Common Crawl*](https://commoncrawl.org/). Most open-source (and many closed-source) language models are trained at least in part on data extracted from the Common Crawl.

### Weights {#weight}

See [parameters](#parameters).

## Open versus Closed {#app:os}

In general, an informational resource --- such as software or data --- is **open** when it is publicly available for free reuse by others and **closed** when it is not. Openness has both practical and legal dimensions. The practical dimension is that the information must actually be available to the public. For example, software is open in this sense when its source code is available to the public. The legal dimension is that the public must have the legal right to reuse the information. This legal dimension is typically ensured with an "open-source" or "public" license that provides any member of the public with the right to use the information.

Open versus closed is not a binary distinction. For one thing, an informational artifact could be practically open but legally closed. For another, there are numerous different licenses, which provide users with different rights. Instead, it is always important to break down the specific ways in which information is open for reuse and the ways in which it is not. The relevant forms of openness are different for different types of information. In this section, we discuss some of the common variations on open and closed datasets, models, and software.

### Closed Dataset {#closed-dataset}

Many [models](#model), including [open models](#open-model), have been trained on non-public [datasets](#datasets). Though a high-level description of the dataset may have been released and some portions of it may indeed be public (e.g. nearly all models are trained on Wikipedia), there is insufficient public information for the dataset to be fully reproduced. For example, there might be very little information available on the [curation and pre-processing](#data-curation) techniques applied, or constituent datasets might be described in general terms such as "books" or "social media conversations" without any detail about the source of these datasets. GPT-4 and PaLM are both examples of models trained on non-public datasets.

### Closed Model {#closed-model}

When a [model](#model) is described as closed, it might mean one of three different things. First, a model might have been described in a technical report or paper, but there is no way for members of the public to access or use the model. This is the most closed a model can be. For example, DeepMind described the Chinchilla model in a blog post and paper but was never made accessible to the public [@hoffmann2022training]. Second, a model's [checkpoint](#checkpoint) may not be publicly available, but the general public may be able to access the model in a limited way via an [API](#api) or a web application (often for a fee and with the requirement they must sign a [Terms of Service](#tos)). For example, OpenAI's GPT-3 and GPT-4 have followed this paradigm. In this case, it might be possible for users to reconstruct some of the model's characteristics, but just as with closed-source software they do not have access to the full details. Third, the model itself may have been publicly released for inspection, but without a license that allows others to make free use of it. Practically, the license restrictions may be unenforceable against individual users, but the lack of an open license effectively prevents others from building major products using the model.

### Closed Software {#closed-source}

Closed-source software is any software where the source code has not been made available to the public for inspection, modification, or enhancement. It is worth noting that closed software can contain [open](#os) components. For example, an overall system might be (semi-)closed if it releases its [model](#open-model), but does not disclose its [dataset](#Closed-Dataset). Many open-source licenses, which were developed before the advent of modern generative-AI systems, do not prevent open-source software from being combined with closed components in this way.

### Open Dataset {#open-dataset}

Saying that a [dataset](#datasets) is "open" or "open-source" can mean one of several things. At the most open end of the spectrum, it can mean that the dataset is broadly available to the public for download and that the code and experimental settings used to create it are entirely open-source. (This is the fullest expression of the idea that making a software artifact open requires providing access to the preferred form for studying and modifying it.) In some cases, a dataset is available for download but the code and exact experimental settings used to create it are not public. In both these situations, use of the dataset is normally governed by an open-source [license](#license). For example, one popular set of benchmark datasets for language models is called SuperGLUE. The licenses for its constituent datasets include [*BSD 2-Clause*](https://people.ict.usc.edu/~gordon/copa.html), [*Creative Commons Share-Alike 3.0*](https://github.com/google-research-datasets/boolean-questions), and the [*MIT License*](https://github.com/rudinger/winogender-schemas/tree/master). In more restrictive cases, the dataset is public, but users must agree to contractual terms of service to access it --- and those terms impose restrictions on how the dataset can be used. Finally, many datasets are public but cost money to access. For example, the [*UPenn Linguistics Data Consortium*](https://www.ldc.upenn.edu/) has a catalog of hundreds of high-quality datasets, but individuals need to be affiliated with a member institution of the consortium to access them.

### Open Model {#open-model}

The machine learning community has described a [model](#model) as "open-source" when a trained [checkpoint](#checkpoint) has been released with a [license](#license) allowing anyone to download and use it, and the software package needed to load the checkpoint and perform inference with it have also been open-sourced. A model can be open-sourced even in cases where details about how the model was developed have not been made public (sometimes, such models are referred to instead as **semi-closed**). For example, the model creators may not have open-sourced the software package for training the model --- indeed, they may have not even publicly documented the training procedure in a technical report. Furthermore, a model being open-source does not necessarily mean the training [data](#datasets) has been made public. However, various pre-existing open communities (including the Open Source Initiative, which maintains the canonical Open Source Definition) have objected to usage by the machine learning community, arguing that it does not capture several of the most important qualities of opennness as it has been understood by the software community for over two decades. These qualities include the freedom to inspect the software, the freedom to use the software for any purpose, and the ability to modifiy the software, and the freedom to distribute modifications to others.

### Open Software {#os}

Open-source software is software with source code that anyone can inspect, modify, and enhance, for any purpose. Typically such software is licensed under a standardized open-source [license](#license), such as the [*MIT License*](https://opensource.org/license/mit/) or the [*Apache license*](https://opensource.org/license/apache-2-0/). Machine-learning systems typically consist of several relatively independent pieces of software; there is the software that builds the [training dataset](#datasets), the software that [trains](#training) the model, and the software that does [inference](#inference) with the [model](#model). Each of these can be independently open-sourced.

## Legal Concepts in Intellectual Property and Software {#gloss:legal-concepts}

### Claims {#claims}

[Patent](#patent) **claims** are extremely precise statements that define the scope of protection within the patent. Patent claims are carefully written to be broad enough to encompass potential variations of the invention and specific enough to distinguish the invention from prior art.

### Copyright {#copyright}

**Copyright** grants exclusive rights to creators of original works. For a work to be copyrightable, it must meet a certain criteria: (1) it must be original, and (2) it must possess a sufficient degree of creativity. Copyright does not protect facts or concepts, but expressions of those ideas fixed in a tangible medium (e.g., the idea for a movie, if not written down or recorded in some way, is typically not copyrightable; a screenplay is typically copyrightable). Copyright laws provide protections for various forms of creative expression, including, but not limited to, literary works, artistic works, musical composition, movies, and software [@copyright].

### Copyright Infringement {#infringe}

Copyright infringement occurs when someone uses, reproduces, distributes, performs, or displays copyrighted materials without permission from the copyright owner. This act breaches the exclusive rights held by the copyright holder.

### Damages {#damages}

In the context of [IP](#ip), **damages** refers to financial compensation awarded to the owner of IP for harms and losses sustained as a result of IP infringement. When IP rights such as [patents](#patent), [copyright](#copyright), or trademarks are violated, the owner of the IP may file a legal claim for damages. These can cover lost profits, reputational damage, or [licensing](#license) fees.

### Fair Use {#fair-use}

**Fair use** is a legal concept that allows limited use of [copyrighted](#copyright) materials without permission from the copyright owner [@fairuse]. Typically, fair use applies to contexts such as teaching, research, and news reporting, and fair use analyses consider the purpose of use, scope, and the amount of material used.

### The Field of Intellectual Property (IP) {#ip}

The **field of Intellectual Property (IP)** refers to a set of laws that grant exclusive rights for creative and inventive works. IP laws protect and promote ideas by providing incentives for innovation and protecting owners of inventions (e.g. written works, music, designs, among others). Intellectual property laws include copyright, patents, trademarks and trade dress, and trade secrets. Intellectual property law is often the first recourse for conflicts around emerging technologies where more specific legislation has not yet crystallized. Property law is well-developed, widely applicable, and carries significant penalties including fines and forced removal or destruction of work.

### Harm {#harm}

Many areas of law only regulate actions that cause some identifiable **harm** to specific victims. For example, people who have not suffered individual harms may not have "standing" to bring a lawsuit in federal court. For another, damage awards and other remedies may be limited to the harms suffered by the plaintiffs, rather than dealing more broadly with the consequences of the defendant's conduct. It is important to note that what counts as a cognizable harm is a legal question, and does not always correspond to people's intuitive senses of when someone has suffered a harm. Physical injury is the most obvious and widely accepted type of legal harm; other widely recognized forms of harm include damage to property, economic losses, loss of liberty, restrictions on speech, and some kinds of privacy violations. But other cases have held that fear of future injury is not a present harm. Thus, having one's personal information included in a data breach may not be a harm by itself --- but out-of-pocket costs and hassle to cancel credit cards are recognized harms.

### Idea vs. Expression {#idea-expression}

This **idea vs. expression** dichotomy gets at the distinction between underlying concepts (or ideas) conveyed by a work, and the specific, tangible manner in which those are expressed. An idea refers to an abstract concept or notion behind a creative work, and ideas are not subject to [copyright](#copyright) protection. However, expressions, as tangible manifestations, are. Tangible fixed expressions of ideas include words, music, code, or art. It is important to note that within copyright law, rights are granted to expression of ideas, not ideas themselves.

See [copyright](#copyright).

### License {#license}

A **license** gives legal permission or authorization, granted by the rights holder to others. License agreements explicitly outline rights that are granted, as well as limitations, restrictions, and other provisions related to its scope, for example, duration. Licenses are common practice within the [field of IP](#ip), and are commonly used in software, music, and film industries.

### Non-Expressive or Non-Consumptive {#non-expressive-or-non-consumptive}

Certain uses of [copyrighted](#copyright) materials can be **non-expressive** or **non-consumptive**. In such cases, copyrighted material is used in a way that does not involve expressing or displaying original work to users. Some examples include text mining, building a search engine, or various forms of computational analyses.

### Patent {#patent}

A **patent** confers exclusive rights to inventors, granting them the authority to prevent others from making, using, or selling their inventions without permission. Patents create incentives for innovation by providing inventors with a time-based protection from the filing date. To obtain a patent, inventions must be new, inventive, and industrially applicable. Creators apply for patents; their applications must contain [claims](#claims) that describe what is novel in the work.

### Prior Art {#prior-art}

**Prior art** is evidence of existing knowledge or information that is publicly available before a certain date. Prior art is critical in adjudicating the novelty and nonobviousness of a new invention and may include other [patents](#patent). Patent examiners search for prior art to determine the patentability of the [claimed](#claims) invention. Further, prior art informs the patent's applicability and scope.

### Terms of Service {#tos}

**Terms of service (ToS)** refers to a contractual agreement between the IP owner and users of [licenses](#license) that govern the use and access to the protected content. ToS outline rights, restrictions, and obligations involved. ToS may specify permitted uses, licensing terms, and how IP may be copied or distributed. ToS safeguard IP owners' rights and ensure compliance with legal standards in the use of IP.

### Transformative Use {#transformative-use}

Expression can build on prior expression. In some cases, a new piece of [copyrightable](#copyright) material may borrow or re-purpose material from prior work. If this new material creates something inventive, new, and substantially different from the original work, then it can be considered **transformative use** of the original work, as opposed to [infringing](#infringe) on the original copyright owner's exclusive rights. The new material may also be copyright eligible. Parody is one common type of transformative use.

## Privacy {#app:privacy}

### Anonymization {#anonymization}

**Anonymization** is the process of removing or modifying personal data in a way that it cannot be attributed to an identifiable individual.

### The California Consumer Privacy Act (CCPA) {#the-california-consumer-privacy-act}

 The **CCPA** is a California state law that provides consumers with the right to know what personal information businesses collect about them, the right to request their personal information be deleted, and the right to opt-out of sales of their personal information. The CCPA applies to all businesses that operate in California, as well as those outside of California that may transfer or process the personal information of California residents.

### Consent {#consent}

**Consent** is the voluntary and informed agreement given by an individual for the collection, use, or disclosure of their personal information. In the context of data, consent often requires clear and specific communication about the purpose and use of collected data.

### Differential Privacy {#differential-privacy}

**Differential privacy** (DP) is an approach for modifying algorithms to protect the membership of a given **record** in a dataset [@dp]. Informally, these guarantees are conferred by adding small amounts of **noise** to the individual data [examples](#examples) in the dataset. Let us say that there are two version of a dataset, $D$ and $D'$, where the former contains an example $E$ and the latter does not. If we were to run differentially private algorithms to compute statistics on the datasets $D$ and $D'$, we would not be able to tell by those statistics which dataset contains $E$ and which does not. As a result, we can no longer use the computed statistics to infer whether or not the original training data contained the example $E$.

Differential privacy is a theoretical framework that encounters some challenges in practice. For example, the amount of noise one must add to data may impact the accuracy of statistics computed, or, when used for generative-AI models, may impact performance of the model. On the other hand, someone using a differentally private approach needs to add enough noise to ensure that the two datasets $D$ and $D'$ cannot be differentiated through computed statistics. Finally, differential privacy was originally created for tabular data and encounters challenges adapting to the unstructured data commonly used for generative-AI models. For more on the challenges of applying differential privacy to language models, please see @brown2022privacy.

### The General Data Protection Regulation (GDPR) {#gdpr}

 **GDPR** is a comprehensive data protection law implemented by the European Union in 2018 [@gdpr]. The GDPR governs the collection, use, storage, and protection of personal data for EU residents. The law sets out specific rights for individuals regarding their personal data, such as the right to access, rectify, and delete their data, as well as the right to know how their data is being processed. Further, the GDPR imposes obligations on organizations such as businesses that handle personal data to ensure that proper data protection measures are in place and that consent is obtained for data processing. Non-compliance results in fines and penalties.

### Personally Identifiable Information (PII) {#pii}

 **Personally Identifiable Information (PII)** refers to data that can be used to identify an individual. PII can include names, addresses, phone numbers, social security numbers, email addresses, financial information, and biometric data. PII is sensitive, and organizations that collect PII are required to implement appropriate measures, adhering to relevant data protection laws (such as the GDPR) to safeguard its confidentiality and integrity.

### Privacy Policy {#privacy-policy}

A **privacy policy** consists of documents that outline how organizations collect, use, store, and protect personal information. Privacy policies are meant to inform individuals about their rights and the organization's data processing practices.

### Privacy Violation {#privacy-violation}

A **privacy violation** involves unauthorized or inappropriate intrusion into an individual's personal information or activities. Privacy violations may occur in various forms from data breaches, surveillance, identity theft, or sharing personal or sensitive information without consent. These violations may lead to significant harm such as the loss of personal autonomy, reputational damage, or financial loss.

### The Right to be Forgotten {#the-right-to-be-forgotten}

Some countries' legal systems recognize a **right to be forgotten** that grants individuals the ability to request the removal of their personal information from online platforms or search-engine results. The idea is that the legitimate public interest in knowing about other people's past conduct can be outweighed when the information about it is out-of-date or misleading. The European Union's [GDPR](#gdpr) includes a form of the right to be forgotten.

### Tort {#tort}

A **tort** is a civil wrongdoing that causes harm or injury to another person or their property. Tort law provides remedies and compensation to individuals who suffer harm as a result of someone else's actions or negligence.

# Metaphors {#app:metaphors}

We briefly discuss several metaphors for Generative AI that came up in the GenLaw discussions. It is worth considering why these metaphors are helpful and where they start to break down.

### Models are trained.

Machine learning practitioners will often say they "train" models. Training brings to mind teaching a dog to perform tricks by enforcing good behavior with treats. Each time the dog performs the desired behavior, they get a treat. As the dog masters one skill it may move onto another. Model training is similar in the sense that models are optimized to maximize some reward[^1]. This "reward" is computed based on how similar the model's outputs are to desired outputs from the model.

However, unlike training a dog, model training does not typically have a curriculum;[^2] there is no progression of easier to harder skills to learn, and the formula for computing the reward remains the same throughout model training.

### Models learn like children do.

"Learning" is the active verb we use to describe what a model does as it is being *trained*--- a model is *trained*, and during this process it *learns*. Model learning is the most common anthropomorphic metaphor applied to machine learning models. The use of the word "learning" by machine learning practitioners has naturally led to comparisons between how models learn and how human children do. Both children and machine learning models "skilled imitators," acquiring knowledge of the world by learning to imitate provided exemplars However, human children and Generative AI obviously use very different mechanisms to learn. Techniques that help generative-AI systems to learn better, such as increasing model size, have no parallels in child development, and mechanisms children use to "extract novel and abstract structures from the environment beyond statistical patterns" have no machine learning comparisons [@yiu2023imitation].

### Generations are collages.

Quoted from @lee2023talkin [p. 58]:

It also may seem intuitively attractive to consider generations to be analogous to collages. However, while this may seem like a useful metaphor, it can be misleading in several ways. For one, an artist may make a collage by taking several works and splicing them together to form another work. In this sense, a generation is not a collage: a generative-AI system does not take several works and splice them together. Instead, as we have described above, generative-AI systems are built with models trained on many data examples. Moreover, those data examples are not explicitly referred back to during the generation process. Instead, the extent that a generation resembles specific data examples is dependent on the model encoding in its parameters what the specific data examples look like, and then effectively recreating them. Ultimately, it is nevertheless possible for a generation to look like a collage of several different data examples; however, it is debatable whether the the process that produced this appearance meets the definition for a collage. There is no author "select[ing], coordinat[ing], or arrang[ing]"[^3] training examples to produce the resulting generation.

### Language models are stochastic parrots.

@bender2021parrots describe a language model as a stochastic parrot, a "system for haphazardly stitching together sequences of linguistic forms it has observed in its vast training data, according to probabilistic information about how they combine, but without any reference to meaning." Like parrots mimicking the sounds they hear around them, language models repeat the phrases they are exposed to, but have no conception of the human meaning behind these phrases. This analogy is useful because it references the very real problem of machine learning models simply outputting their most frequent training data. Critics of the stochastic parrot analogy say that it undervalues the competencies that state-of-the-art language models have. Some critics take this further and say that these competencies imply models understand meaning in a human-like way [@piantadosi2022meaning].[^4] For example, proponents of this analogy might argue that Generative AI passing a difficult standardized exam (such as the Bar Exam [@katz2023gpt] or the GRE [@gpt4]) is more about parroting training data than human-like skill.

### Language models are noisy search engines.

A search engine allows users to search for information within a large database using natural language queries. Like a search engine, language models also return information in response to a natural language query. However, while a search engine queries the entries in its database and returns the most appropriate ones, a language model does not have direct access to its training data and can only make predictions based on the information stored in the model weights.[^5] Most often the output will be a mixture of information contained in many database entries. Some model outputs may quote directly from relevant entries in the database (in the case of [memorization](#memorization)), but this is not reflective of the most typical outputs.

Sometimes generations from a language model will convey similar information that one might learn from running a search; however, sometimes it will not because the underlying [algorithm](#algorithm) is different. Thus, while some generations answer the prompt in a similar way to a search, we can more generally think of generative-model outputs as a noisy version of what is actually in the database. Currently, such outputs also tend to lack attribution to the original data entries, and sometimes are incorrect.

[^1]: Maximizing a reward is exactly equivalent to minimizing a loss     (except for the extra minus sign), but due to historical reasons,     machine learning practitioners use the latter phrasing more often.

[^2]: Curriculum learning is an entire field of research in machine     learning, but it is not currently standard to use a curriculum.

[^3]: § 101 (definition of "compilation").

[^4]: Whether models are human-like, or the outputs are simply "really     good" is less pertinent for how generations and inputs should be     regulated.

[^5]: The training data is seen during training, but models are used     separately from the training data.

## References