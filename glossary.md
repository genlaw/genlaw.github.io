--- 
title: "Glossary" 
tags: [machine learning, datasets, generative ai, artificial intelligence, copyright, privacy, law] 
return-genlaw: true 
return-footer: true 
---

We roughly divide our glossary into sections:

- [Machine Learning and Generative AI](#gloss:machine-learning)
- [Open versus Closed](#app:os)
- [Legal Concepts in Intellectual Property and Software](#gloss:legal-concepts)
- [Privacy](#app:privacy)

See also: [resources](resources.html), [metaphors](#metaphors.html), and the [GenLaw Report](2023-report.html)

# Concepts in Machine Learning and Generative AI {#gloss:machine-learning}

## Algorithm {#algorithm}

An **algorithm** is a formal, step-by-step specification of a process. Machine learning uses algorithms for **[training](glossary.html#training)** **[models](glossary.html#model)** and for applying models (a process called **[inference](glossary.html#inference)**). In training, the algorithm takes a model **[architecture](glossary.html#architecture)**, **[training data](glossary.html#datasets)**, **[hyperparameters](glossary.html#hyperparameter)**, and a random seed (to enable random choices during statistical computations) to produce trained model **[parameters](glossary.html#parameters)**.

In public discourse around social media, the term algorithm is often used to refer to methods for optimizing the probability that a user will engage with a post; however, it is important to note that algorithms describe many processes including, for example, the process of sorting social media posts by date.

## Alignment {#alignment}

**Alignment** refers to the process of taking a **[pre-trained](glossary.html#pre-training-and-fine-tuning)** **[model](glossary.html#model)** and further tuning it so that its outputs are *aligned* with a policy set forth by the model developer. Alignment can also refer to the *state of being aligned* --- some academic papers might compare and contrast between an aligned and an unaligned model. The goals included in an alignment policy (sometimes called a constitution) vary from developer to developer, but common ones include:

-   Following the intent of user-provided instructions;
-   Abiding by human values (e.g., not emitting swear-words);
-   Being polite, factual, or helpful;
-   Avoiding generating copyrighted text.

While the goals of alignment are vast and pluralistic, current techniques for achieving better alignment are broadly applicable across goals. These techniques include **[reinforcement learning](glossary.html#reinforcement-learning)** with human feedback and full model **[fine-tuning](glossary.html#pre-training-and-fine-tuning)**. Specifying the desired properties of alignment often requires a special dataset. These datasets may include user-provided feedback, supplied through the user interface in a generative-AI product.

## Application Programming Interface (API) {#api}

Companies choose between releasing generative-AI **[model](glossary.html#model)** functionality in a variety of ways: They can release the model directly by **[open-sourcing](glossary.html#open-model)** it; they can embed the model in a software system, which they release as a product; or, they can make the model (or the system its embedded in) available via an **Application Programming Interface (API)**. When a model is open-source, anyone can take the **[checkpoint](glossary.html#checkpoint)** and load it onto a personal computer in order to use it for **[generation](glossary.html#generation)** (by embedding the checkpoint in a program). In contrast, when a company only makes their generative-AI model or system available via an API, that means that users access it in code. The user writes a query in a format specified by the company, then sends the query to the company's server. The company then runs the model or system on their own computers, and provides a response to the user with the generated content. API access usually requires accepting the company's **[Terms of Service](glossary.html#terms-of-service)**, and companies may add extra layers of security on top of the model (such as rejecting queries identified as being in violation of the ToS).

## Architecture {#architecture}

The design of a **[model](glossary.html#model)** is called its **architecture**. For a **[neural network,](glossary.html#neural-network)** architectural decisions include the format of inputs that can be accepted (e.g., images with a certain number of pixels), the number of layers, how many **[parameters](glossary.html#parameters)** per layer, how the parameters in each layer are connected to each other, and how we represent the intermediate state of the model as each layer transforms input to output. The most common architecture for language tasks is called a Transformer [@vaswani2017attention], for which there are many variations.

Many contemporary models appear in **model families** that have similar architectures but different sizes, often differentiated by the total number of parameters in the model. For example, Meta originally released four sizes of the LLaMA family that had almost the exact same architectures, differing only in the number of layers and size of the intermediate **[vector representations](glossary.html#vector-representation)**. More layers and wider internal representations can improve the capability of a model, but can also increase the amount of time it takes to **[train](glossary.html#training)** the model or to do **[inference.](glossary.html#inference)**

## Attention {#attention}

An **attention** mechanism is a sub-component of a **[transformer](glossary.html#transformer)** architecture. This mechanism allows a **[neural network](glossary.html#neural-network)** to selectively focus on (i.e., **attend** to) specific tokens in the input sequence by assigning different attention weights to each token. Attention also refers to the influence an input token has on the token that is generated by the model.

## Base Model {#base-model}

A **base model**, which is sometimes called a "**[foundation](glossary.html#foundation-model)**" or "pre-trained model," is a **[neural network](glossary.html#neural-network)** that has been **[pre-trained](glossary.html#pre-training-and-fine-tuning)** on a large, general-purpose dataset (e.g., a **[web crawl](glossary.html#web-crawl)**). Base models can be thought of as the first step of training and a good building block for other models. Thus, base models are not typically exposed to Generative AI users; instead they are adapted to be more usable either through **[alignment](glossary.html#alignment)** or **[fine-tuning](glossary.html#pre-training-and-fine-tuning)** or performing **[in-context learning](glossary.html#in-context)** for *specific* tasks. For example, OpenAI trained a base model called GPT-3 then adapted it to follow natural-language instructions from users to create a subsequent model called InstructGPT [@instructgpt].

See also: **[pre-training and fine-tuning](glossary.html#pre-training-and-fine-tuning)**.

## Checkpoint {#checkpoint}

While a **[model](glossary.html#model)** is being trained, all of its **[parameters](glossary.html#parameters)** are stored in the computer's memory, which gets reset if the program terminates or the computer turns off. To keep a model around long-term, it is written to long-term memory, i.e., a hard drive in a file called a **checkpoint.** Often, during training, checkpoints are written to disk every several thousand steps of training. The minimum bar for a model to be considered **[open-source](glossary.html#open-model)** is if there has been a public release of one of its checkpoints, as well as the code needed to load the checkpoint back into memory.

## Context Window {#context-window}

Also called **prompt length**. Generative AI **[prompts](glossary.html#prompt)** typically have a fixed **context window**. This is the maximum accepted input length for the model and arises because models are trained with data examples that are no longer than this maximum context window. Inputs longer than this maximum context window may result in generations with performance degradations.

## Data Curation and Pre-Processing {#data-curation-and-pre-processing}

**Data curation** is the process of creating and curating a dataset for training a model. In the past, when datasets were smaller, they could be manually curated, with human annotators assessing the quality of each example. Today, the final datasets used to train generative machine-learning models are typically automatically curated. For example, data examples identified as "toxic" or "low quality" may be removed. This filtering is done using an **[algorithm](glossary.html#algorithm)**. These algorithms may include heuristic rules (e.g., labeling examples containing forbidden words as toxic) or may use a different machine-learning model trained to classify training examples as low-quality or toxic. Another common curation step is to deduplicate the dataset by identifying examples that are very similar to each other and removing all but one copy [@lee2022dedup].

**Data pre-processing** involves transforming each example in the dataset to a format that is useful for the task we want our machine-learning model to be able to do. For example, a web page downloaded from a **[web crawl](glossary.html#web-crawl)** might contain HTML markup and site navigation headers. We may want to remove these components and keep only the content text.

Data curation and pre-processing happens directly to the data and are independent of any model. Different models can be trained on the same dataset. For more information about dataset curation and pre-processing, see @lee2023explainers [Chapter 1]

## Datasets {#datasets}

**Datasets** are collections of data **[examples](glossary.html#examples)**. Datasets are used to **[train](glossary.html#training)** machine-learning models. This means that the **[parameters](glossary.html#parameters)** of a machine-learning model depend on the dataset. Different machine-learning models may require different types of data and thus different datasets. The choice of dataset depends on the task we want the machine-learning model to be able to accomplish. For example, to train a model that can generate images from natural-language descriptions, we would need a dataset consisting of aligned image-text pairs.

Datasets are often copied and reused for multiple projects because (a) they are expensive and time-consuming to create, and (b) reuse makes it easier to compare new models and algorithms to existing methods (a process called **benchmarking**). Datasets are usually divided into **[training](glossary.html#training)** and **testing** portions. The testing portion is not used during training, and is instead used to measure how well the resulting model **[generalizes](glossary.html#generalization)** to new data (how well the model performs on examples that look similar to the training data but were not actually used for training).

Datasets can be either created directly from a raw source, such as Wikipedia or a **[web crawl](glossary.html#web-crawl)**, or they can be created by assembling together pre-existing datasets. Here are some popular "source" datasets used to train generative machine learning models:

-   [*Wikipedia*](https://huggingface.co/datasets/wikipedia)
-   [*Project Gutenberg*](https://arxiv.org/abs/1812.08092v1)
-   [*Common Crawl*](https://commoncrawl.org/)
-   [*C4*](https://github.com/allenai/allennlp/discussions/5056)
-   [*ImageNet*](https://www.image-net.org/)
-   [*LAION-400M*](https://laion.ai/blog/laion-400-open-dataset/)
-   [*ROOTS*](https://huggingface.co/papers/2303.03915)

Here are some popular datasets that were created by collecting together several pre-existing datasets:

-   The Pile (The Pile was formerly listed online, but was removed     following the Huckabee v. Meta Platforms, Inc., Bloomberg L.P.,     Bloomberg Finance, L.P., Microsoft Corporation, and The EleutherAI     Institute class action complaint [@huckabee]. Notably, a     de-duplicated version of the dataset is still available on     HuggingFace via EleutherAI, as of the writing of this     report [@deduppile].)
-   [*Dolma*](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64)
-   [*RedPajama*](https://github.com/togethercomputer/RedPajama-Data)

Collection-based datasets tend to have a separate license for each constituent dataset rather than a single overarching license.

## Decoding {#decoding}

A **decoding algorithm** is used by a **[language model](glossary.html#language-model)** to generate the next word given the previous words in a prompt. There are many different types of decoding algorithms including: greedy algorithm, beam-search, and top-k [@decoding]. For language models, the term **decoding** may be used interchangeably with **[inference](glossary.html#inference)**, **[generation](glossary.html#generation)**, and **sampling**.

## Diffusion-Based Modeling {#diffusion-based-modeling}

**Diffusion-based modeling** is an algorithmic process for model training. Diffusion is *not* itself a **[model architecture](glossary.html#architecture)**, but describes a process for training a model architecture (typically, an underlying **[neural network](glossary.html#neural-network)**) [@sohldickstein2015dpm; @rombach2022diffusion; @ho2020denoising; @song2019grads]. Diffusion-based models are commonly used for image generation, such as in Stable Diffusion text-to-image models [@stable-diffusion]. **Diffusion probabilistic model**, **diffusion model**, and **latent diffusion model** have become terms that refer to types of models (typically, neural networks) that are trained using diffusion processes.

## Embedding {#embedding}

(Also called **vector representation**.) **Embeddings** are numerical representations of data. There are different types of embeddings, such as word embeddings, sentence embeddings, image embeddings, etc. [@murphy2022mlintro p. 26, pp. 703-10]. Embeddings created with machine-learning models seek to model statistical relationships between the data seen during training.

"Common embedding strategies capture semantic similarity, where vectors with similar numerical representations (as measured by a chosen distance metric) reflect words with similar meanings.

Needless to say, such quantified data are not identical to the entities they reflect, *however*, they can capture certain useful information about said entities" Quoted from @lee2023talkin [p. 7]:

## Examples {#examples}

An **example** is a self-contained piece of data. Examples are assembled into **[datasets](glossary.html#datasets)**. Depending on the dataset, an example can be an image, a piece of text (such as content of a web page), a sound snippet, a video, or some combination of these. Often times examples are **labeled** --- they have an input component that can be passed into a machine-learning model, and they have a target output, which is what we would like the model to predict when it sees the input. This format is usually referred to as an "input-output" or "input-target" pair. For example, an input-target example pair for training an image-classification model would consist of an image as the input, and a label (e.g. whether this animal is a dog or a cat) as the target.

## Generalization {#generalization}

**Generalization** in machine learning refers to a model's ability to perform well on unseen data, i.e., data it was not exposed to during training. **Generalization error** is usually measured evaluating the model on **training** data and comparing it with the evaluation of the model on **test** data [@generalization]. (See **[datasets](glossary.html#datasets)** for more on training and test data.)

## Generation {#generation}

**Generative models** produce complex, human-interpretable outputs, such as full sentences or natural-looking images, called **generations**. Generation also refers to the process of applying the generative-AI model to an input and generating an output. The input to a generative model is often called a **[prompt](glossary.html#prompt)**. More traditional machine-learning models are limited to ranges of numeric outputs (**regression**) or discrete output labels like "cat" and "dog" (**classification**). More commonly the word **[inference](glossary.html#inference)** is used to describe applying a traditional machine-learning model to inputs.

Generative models can output many different generations for the same prompt, which all may be valid to a user. For example, there may be many different kinds of cats that would all look great wearing a hat. This makes evaluating the performance of generative models challenging (i.e., how can we tell which is the "best" such cat?). Recent developments in Generative AI have significantly increased generation quality. However, even high-quality generations can reach an **uncanny valley** with subtly wrong details (e.g., incorrect number of fingers on a typical human hand).

See also: **[inference](glossary.html#inference)**.

## Fine-Tuning {#fine-tuning}

See description alongside **[pre-training](glossary.html#pre-training-and-fine-tuning)**.

## Foundation Model {#foundation-model}

**Foundation model** is a term coined by researchers at Stanford University [@bommasani2021foundation] to refer to neural networks that are trained on very large general-purpose datasets, after which they can be adapted to many different applications. Another common word, used interchangeably with foundation model, is **[base model](glossary.html#base-model)**.

## Hallucination {#hallucination}

There are two definitions of the word **hallucination** [@hallucination]. First, hallucination can refer to generation that does not accord with our understanding of reality, e.g., the generation for the text prompt "bibliographies" consisting of research papers that do not exist. These hallucinations may occur because generative models do not have explicit representations of facts or knowledge. Second, generations that have nothing to do with the input may also be termed a hallucination, e.g., an image generated of a cat wearing a hat in response to the prompt "Science fiction from the 80s."

## Hyperparameter {#hyperparameter}

**Hyperparameters** are settings for the model or training process that are manually specified prior to training. Unlike **[parameters](glossary.html#parameters)**, they are not typically learned (often, they are hand-selected). Examples of hyperparameters for a model include properties of the architecture, such as input sequence length, the number of model parameters per **layer**, and number of layers. Examples of hyperparameters that determine the behavior of the training algorithm include the learning rate, which controls how much we update model parameters after each input/output training example (i.e., the magnitude of the update). The process of picking hyperparameters is typically called **hyperparameter optimization**, which is its own field of research.

## In-Context Learning (Zero-Shot / Few-Shot) {#in-context}

A **[base model](glossary.html#base-model)** can be used directly without creating new **[checkpoints](glossary.html#checkpoint)** through **[fine-tuning](glossary.html#pre-training-and-fine-tuning)**. **In-context learning** is a method of adapting the model to a specific application by providing additional **[context](glossary.html#context-window)** to the model through the **[prompt](glossary.html#prompt)**. This can be used instead of the more computationally expensive **[fine-tuning](glossary.html#pre-training-and-fine-tuning)** process, though it may not be as effective. In-context learning involves creating an input or **[prompt](glossary.html#prompt)** to a model in a way that constrains a desired output. Typically the input includes either **instructions** (**zero-shot**: a natural language description of the output) or a small number of **examples** of input-output pairs (**few-shot**). "**Shot**" refers to the number of examples provided.

## Inference {#inference}

More traditional machine-learning methods (like **regression** and **classification**) often use the word **inference** instead of **[generation](glossary.html#generation)** to refer to the process of applying a trained model to input data. However, both **inference** and **generation** are used to describe the generation process for a generative-AI model.

See also: **[generation](glossary.html#generation)**.

## Language Model {#language-model}

A **language model (LM)** is a type of **[model](glossary.html#model)** that takes a sequence of text as input and returns a prediction for what the next word in the text sequence should be. This prediction is usually in the form of a probability distribution. For example, when passed the input "It is raining," the language model might output that the probability of "outside" is 70%, and the probability of "inside" is 5%. Language models used to be entirely statistical; the probability of a "outside" coming next in the phrase would be computed by counting the number of times "outside" occurred after the sequence "It is raining" in their training **[dataset](glossary.html#datasets)**. Modern language models are implemented using **[neural networks](glossary.html#neural-network)**, which have the key advantage that they can base their output probabilities on complex relationships between sequences in the training data, rather than just counting how often each possible sequence occurs. As an illustrative example, a neural network may be able to use information from a phrase like "It's a monsoon outside" occurring in the training data to increase the probability of the word "outside."

A language model can be used for **[generation](glossary.html#generation)** by employing an **[algorithm](glossary.html#algorithm)** that selects a word to generate given the probabilities output by the model for some prompt sequence. After each word is selected, the algorithm appends that word to the previous **[prompt](glossary.html#prompt)** to create a new prompt, which is then used to pick the next word, and so on. Such an algorithm is referred to as a **[decoding algorithm](glossary.html#decoding)**.

Language-model generation can be used to implement many tasks, including autocomplete (given the start of a sentence, how should the sentence be completed?) and translation (translate a sentence from English to Chinese). The probabilities output by the language model can also be used directly for tasks. For example, for a sentiment classification task, we might ask the language model whether "but" or "because" is a more probable next word given the prompt "the food was absolutely delicious."

For more information, see @riedl2023transformers.

## Large Language Model (LLM) {#llm}

The term **large language model (LLM)** has become popular as a way to distinguish older **[language models](glossary.html#language-model)** from more modern ones, which use the **[transformer](glossary.html#transformer)** architecture with many parameters and are trained on web-scale datasets. Older models used different **[model architectures](glossary.html#architecture)** and fewer **[parameters](glossary.html#parameters)**, and were often trained on smaller, more narrowly scoped datasets. Consensus for what constitutes "large" has shifted over time as previous generations of large language models are replaced with models with even more parameters that are trained for even more steps.

## Loss {#loss}

**[Neural networks](glossary.html#neural-network)** (and other models) take an input and predict an output. The distance (measured by some specified function) between this output and the output we *expect* the model to predict (i.e., the target) is called the **loss**. Neural networks are **[trained](glossary.html#training)** by an **[algorithm](glossary.html#algorithm)** that repeatedly passes **[examples](glossary.html#examples)** into the the network, measures the loss compared to the expected output, and then updates the network's **[parameters](glossary.html#parameters)** so as to reduce the size of the loss on these examples. The goal of this training process is to minimizing the loss over all the exampling in the training dataset.

Some research areas (for example, **[reinforcement learning](glossary.html#reinforcement-learning)**) refer to "maximizing a reward" rather than "minimizing a loss." These concepts are largely interchangeable; a loss can be turned into a reward by adding a negative sign to it.

See also: **[objective](glossary.html#objective)**.

## Memorization {#memorization}

**Memorization** generally refers to being able to deduce or produce a **[model's](glossary.html#model)** given training **[example](glossary.html#examples)**.

There are further delineations in the literature about different types of memorization. A training **[example](glossary.html#examples)** may be **memorized** by a model if information about that training example can be **discovered** inside the model through any means. A training example is said to be **extracted** from a model if that model can be prompted to generate an output that looks exactly or almost exactly the same as the training example. A training example may be **regurgitated** by the model if the generation looks very similar or almost exactly the same as the training example (with or without the user's intention to extract that training example from the model).

To tease these words apart: a training example is **memorized** by a model and can be **regurgitated** in the generation process regardless of whether the intent is to **extract** the example or not.

The word memorization itself may be used to refer to other concepts that we may colloquially understand as "memorization." For example, facts and style (artists style) may also be memorized, regurgitated, and extracted. However, this use should not be confused with technical words (e.g., **extraction**) with precise definitions that correspond to metrics.

## Model {#model}

**Models** are at the core of contemporary machine learning. A model is a mathematical tool that takes an **input** and produces an **output**. A simple example might be a model that tells you whether a temperature is above or below average for a specific geographic location. In this case the input is a number (temperature) and the output is binary (above/below).

There could be many versions of this model depending on geographic location. The behavior of the model is defined by an internal **[parameter](glossary.html#parameters)** (or, **[weight](glossary.html#weights)**). In our temperature example, the model has one parameter, the average temperature for the location. The process for setting the value of the parameter for a specific version of the model is called **[training](glossary.html#training)**. In this case we might train a model for New York City by gathering historical temperature data and calculating the average. The process of gathering historical data is called **[data collection](glossary.html#data-curation-and-pre-processing)**. The process of training --- in this case, calculating the average --- is an **[algorithm](glossary.html#algorithm)**.

A saved copy of a model's trained parameters is called a **[checkpoint](glossary.html#checkpoint)**. We might save separate checkpoints for different cities or save new checkpoints for our New York City model if we retrain with new data. The process of applying the model to new inputs is called **[inference](glossary.html#inference)**. To create an output for a temperature, we also apply an algorithm: subtract the parameter from the input, and return "above" if the difference is positive.

Our temperature example is a very simple model. In machine learning, models can be arbitrarily complex. A common type of model **[architecture](glossary.html#architecture)** is a **[neural network](glossary.html#neural-network)**, which (today) can have billions of parameters.

It is important to note that models are often embedded within **software systems** that can get deployed to public-facing users. For example, GPT-4 is a generative-AI model that is embedded within the ChatGPT system, which also has a user interface, developer **[APIs](glossary.html#api)**, and other functionality, like **input and output filters**. The other components are not part of the model itself, but can work in concert with the model to provide overall functionality.

## Multimodal {#multimodal}

Generative-AI models may generate content in one **modality** (text, images, audio, etc.) or in **multiple modalities**. For example, DALL-E is a multimodal model that transforms text to images.

## Neural Network {#neural-network}

A **neural network** is a type of **[model architecture](glossary.html#architecture)**. Neural networks consist of **layers**, where the output of one layer is used as the input of the next layer. Each layer consists of a set of classifiers (**neurons**) that each performs a simple operation independently of one another. A neural-network model synthesizes multiple simple decisions by passing the input through a series of intermediate transformations. The outputs of all classifiers at layer $n$ are then passed to each classifier in layer $n+1$, and so forth. Each classifier in each layer has **[parameters](glossary.html#parameters)** that define how it responds to input.

## Objective {#objective}

**Objective** is a term of art in machine learning that describes a mathematical function for the **reward** or **[loss](glossary.html#loss)** of a model. A common point of confusion is between the **goal** a model creator might have and the **objective** for a model. The goal of model training is typically to create models that **[generalize](glossary.html#generalization)** well. But that goal (creating models that generalize well) is not a mathematical function that can be maximized. As an example, the training objective for **[transformer](glossary.html#transformer)** models is typically to generate the same next token as in the training data.

## Parameters {#parameters}

**Parameters** are numbers that define the specific behavior of a **[model](glossary.html#model)**. For example, in the linear equation model $y=mx+b$, there are two parameters: the slope $m$ and the **bias** (or, **intercept**) $b$. A more complex example might be a model that predicts the probability a person makes a bicycle trip given the current temperature and rainfall. This could have two parameters: one representing the effect of temperature and one representing the effect of rainfall. Contemporary **[neural network](glossary.html#neural-network)** models have millions to billions of parameters. Model parameters are often interchangeably referred to as model **[weights](glossary.html#weights)**. The values of parameters are saved in files called **[checkpoints](glossary.html#checkpoint)**.

For more on the distinction between **parameters** and **hyperparameters**, see **[hyperparameters](glossary.html#hyperparameter)**.

## Pre-Processing {#pre-processing}

See **[data curation and pre-processing](glossary.html#data-curation-and-pre-processing)**.

## Pre-Training and Fine-Tuning {#pre-training-and-fine-tuning}

Current protocols divide **[training](glossary.html#training)** into a common **pre-training** phase that results in a general-purpose or **[base model](glossary.html#base-model)** (sometimes called a **[foundation](glossary.html#foundation-model)** or pre-trained model) and an application-specific **fine-tuning** phase that adapts a pre-trained model **[checkpoint](glossary.html#checkpoint)** to perform a desired task using additional data. This paradigm has become common over the last five years, especially as **[model architectures](glossary.html#architecture)** have become larger and larger. This is because, relative to pre-training datasets, fine-tuning datasets are smaller, which make fine-tuning faster and less expensive. In general, it is much cheaper to fine-tune an existing base model for a particular task than it is to train a new model from scratch.

As a concrete example, fine-tuning models to "follow instructions" has become an important special case with the popularity of ChatGPT (this is called **instruction tuning**). Examples of interactions in which someone makes a request and someone else follows those instructions are relatively rare on the Internet compared to, for example, question-and-answer forums. As a result, such datasets are often constructed specifically for the purpose of **[LLM](glossary.html#llm)** fine-tuning, and may provide substantial practical benefits for commercial companies.

Because pre-trained models are most useful if they provide a good basis for many distinct applications, model builders have a strong incentive to collect as much pre-training data from as many distinct sources as possible. Fine-tuning results in a completely new model checkpoint (potentially gigabytes of data that must be loaded and served separately from the original model), and tends to require hundreds to thousands of application-specific examples.

However, the distinction between pre-training and fine-tuning is not well-defined. Models are often trained with many (more than two) training stages. For example, the choice to call the first two of, say three, training stages pre-training and the last stage fine-tuning is simply a choice. Finally, pre-training should not be confused with **[data curation or pre-processing](glossary.html#data-curation-and-pre-processing)**.

## Prompt {#prompt}

Most generative-AI systems take as input (currently, this is often some text), which is then used to condition the output. This input is called the **prompt**.

## Regurgitation {#regurgitation}

See **[memorization](glossary.html#memorization)**.

## Reinforcement Learning {#reinforcement-learning}

**Reinforcement learning (RL)** is a type of machine learning for incorporating feedback into systems. For generative models, RL is commonly used to incorporate **human feedback (HF)** about whether the **[generations](glossary.html#generation)** were "good" or "useful," which can then be used to improve future generations. For example, ChatGPT collects "thumbs-up" and "thumbs-down" feedback on interactions in the system. User feedback is just one way of collecting human feedback. Model creators can also pay testers to rate generations.

## Reward {#reward}

See **[loss](glossary.html#loss)**.

## Scale {#scale}

Machine-learning experts use the term **scale** to refer to the number of **[parameters](glossary.html#parameters)** in their model, the size of their training datasets (commonly measured in terms of number of **[examples](glossary.html#examples)** or storage size on disk), or the computational requirements to train the model. The scale of the model's parameter count and the training dataset size directly influence the computational requirements of training. The scale of computation needed for training can be measured in terms of the number of GPU-hours (on a given GPU type), the number of computers/GPUs involved in total, or the number of FLOPs (floating point operations).

Machine-learning practitioners will sometimes talk about **scaling up** a model. This usually means figuring out a way to increase one of the properties listed above. It can also mean figuring out how to increase the fidelity of training examples, e.g., training on longer text sequences or on higher-resolution images.

## Supply Chain {#supply-chain}

Generative-AI systems are created, deployed, and used in a variety of different ways. Other work has written about how it is useful to think of these systems in terms of a **supply chain** that involves many different stages and actors. For more information about the generative-AI supply chain, please see @lee2023talkin.

## Tokenization {#tokenization}

For **[language models](glossary.html#language-model)**, a common **[pre-processing](glossary.html#data-curation-and-pre-processing)** step is to break documents into segments called **tokens**. For example, the input "I like ice cream." might be tokenized into ["I", "like", "ice", "cream", "."]. The tokens can then be mapped to entries in a **vocabulary**. Each entry, or token, in the vocabulary is given an ID (a number representing that token). Each token in the vocabulary has a corresponding **[embedding](glossary.html#embedding)** that is a learned **[parameter](glossary.html#parameters)**. The embedding turns words into numeric representations that can be interpreted and modified by a model.

Each model family tends to share a vocabulary, which is optimized to represent a particular training corpus. Most current models use **subword tokenization** to handle words that would otherwise not be recognized. Therefore, a rare or misspelled word might be represented by multiple tokens, for example ["seren", "dipity"] for "serendipity." The number of tokens used to represent an input is important because it determines how large the effective **[context window](glossary.html#context-window)** of a model is.

Tokens are also used for other modalities, like music. Music tokens may be **semantic tokens** that may be created using another **[neural network](glossary.html#neural-network)**.

## Transformer {#transformer}

A **transformer** is a popular **[model architecture](glossary.html#architecture)** for image, text, and music applications, and the transformer architecture underlies models like ChatGPT, Bard, and MusicLM. An input (text or image) is broken into segments (word **[tokens](glossary.html#tokenization)** or image patches) as a **[pre-processing](glossary.html#data-curation-and-pre-processing)** step. These input segments are then passed through a series of layers that generate **[vector representations](glossary.html#vector-representation)** of the segments. The model has trainable parameters that determine how much **[attention](glossary.html#attention)** is paid to parts of the input. Like many other generative-AI models, the transformer model is trained with an **[objective](glossary.html#objective)** that rewards reproducing a target training **[example](glossary.html#examples)**.

## Training {#training}

Machine-learning **[models](glossary.html#model)** all contain **[parameters](glossary.html#parameters)**. For **[neural networks](glossary.html#neural-network)**, these parameters are typically initialized to random numbers when the network is first created. During an **[algorithmic](glossary.html#algorithm)** process called **training**, these parameters are repeatedly updated based on the training data within the **[training dataset](glossary.html#datasets)** that the model has seen. Each update is designed to increase the chance that when a model is provided some input, it outputs a value close to the target value we would like it to output. By presenting the model with all of the **[examples](glossary.html#examples)** in a dataset and updating the parameters after each presentation, the model can become quite good at doing the task we want it to do.

A common algorithm for training neural network models is **stochastic gradient descent,** or SGD. Training datasets are often too large to process all at once, so SGD operates on small **batches** of dozens to hundreds of examples at a time. Upon seeing a training example, the algorithm generates the model's output based on the current setting of the parameters and compares that output to the desired output from the training data. In the case of a **[language model](glossary.html#language-model)**, we might ask: did we correctly choose the next word? If the output did not match, the algorithm works backwards through the model's layers, modifying the model's parameters so that the correct output becomes more likely. This process can be thought of as leaving "echoes" of the training examples encoded in the parameters of the model.

## Vector Representation {#vector-representation}

See **[embedding](glossary.html#embedding)**.

## Web Crawl {#web-crawl}

A **web crawl** is a catalog of the web pages accessible on the Internet. It is created by a **web crawler**, an **[algorithm](glossary.html#algorithm)** that systematically browses the Internet, trying to reach every single web page (or a specified subset). For example, Google Search functions using a web crawl of the Internet that can be efficiently queried and ranked. Web crawls are very expensive and compute-intensive to create, and so most companies keep their crawls private. However, there is one public web crawl, called [*Common Crawl*](https://commoncrawl.org/). Most open-source (and many closed-source) **[language models](glossary.html#language-model)** are trained at least in part on data extracted from the Common Crawl.

## Weights {#weights}

See **[parameters](glossary.html#parameters)**.

# Open versus Closed Software {#app:os}

In general, an informational resource --- such as software or data --- is **open** when it is publicly available for free reuse by others and **closed** when it is not. Openness has both practical and legal dimensions. The practical dimension is that the information must actually be available to the public. For example, software is open in this sense when its source code is available to the public. The legal dimension is that the public must have the legal right to reuse the information. This legal dimension is typically ensured with an "open-source" or "public" license that provides any member of the public with the right to use the information.

Open versus closed is not a binary distinction. For one thing, an informational artifact could be practically open but legally closed. For another, there are numerous different licenses, which provide users with different rights. Instead, it is always important to break down the specific ways in which information is open for reuse and the ways in which it is not. The relevant forms of openness are different for different types of information. In this section, we discuss some of the common variations on open and closed datasets, models, and software.

## Closed Dataset {#closed-dataset}

Many **[models](glossary.html#model)**, including **[open models](glossary.html#open-model)**, have been trained on non-public (i.e., **closed**) **[datasets](glossary.html#datasets)**. Though a high-level description of the dataset may have been released and some portions of it may indeed be public (e.g. nearly all **[large language models](glossary.html#llm)** are trained on Wikipedia), there is insufficient public information for the dataset to be fully reproduced. For example, there might be very little information available on the **[curation and pre-processing](glossary.html#data-curation-and-pre-processing)** techniques applied, or constituent datasets might be described in general terms such as "books" or "social media conversations" without any detail about the source of these datasets. GPT-4 and PaLM are both examples of models trained on non-public datasets.

## Closed Model {#closed-model}

When a **[model](glossary.html#model)** is described as **closed**, it might mean one of three different things. First, a model might have been described in a technical report or paper, but there is no way for members of the public to access or use the model. This is the most closed a model can be. For example, DeepMind described the Chinchilla model in a blog post and paper, but the model was never made accessible to the public [@hoffmann2022training]. Second, a model's **[checkpoint](glossary.html#checkpoint)** may not be publicly available, but the general public may be able to access the model in a limited way via an **[API](glossary.html#api)** or a web application (often for a fee and with the requirement they must sign a **[Terms of Service](glossary.html#terms-of-service)**). For example, OpenAI's GPT-3.5 and GPT-4 have followed this paradigm. In this case, it might be possible for users to reconstruct some of the model's characteristics, but just as with **[closed-source software](glossary.html#closed-software)**, they do not have access to the full details. Third, the model itself may have been publicly released for inspection, but without a **[license](glossary.html#license)** that allows others to make free use of it. Practically, the license restrictions may be unenforceable against individual users, but the lack of an open license effectively prevents others from building major products using the model.

## Closed Software {#closed-software}

**Closed-source software** is any software where the source code has not been made available to the public for inspection, modification, or enhancement. It is worth noting that closed software can contain **[open](glossary.html#open-software)** components. For example, an overall system might be **semi-closed** if it releases its **[model](glossary.html#open-model)**, but does not disclose its **[dataset](glossary.html#closed-dataset)**. Many open-source **[licenses](glossary.html#license)**, which were developed before the advent of modern generative-AI systems, do not prevent **[open-source software](glossary.html#open-software)** from being combined with closed components in this way.

## Open Dataset {#open-dataset}

Saying that a **[dataset](glossary.html#datasets)** is **open** or **open-source** can mean one of several things. At the most open end of the spectrum, it can mean that the dataset is broadly available to the public for download and that the code and experimental settings used to create it are entirely open-source. (This is the fullest expression of the idea that making a software artifact open requires providing access to the preferred form for studying and modifying it.) In some cases, a dataset is available for download but the code and exact experimental settings used to create it are not public. In both these situations, use of the dataset is normally governed by an open-source **[license](glossary.html#license)**.

For example, one popular set of benchmark datasets for **[large language models](glossary.html#llm)** is called SuperGLUE. The licenses for its constituent datasets include [*BSD 2-Clause*](https://people.ict.usc.edu/~gordon/copa.html), [*Creative Commons Share-Alike 3.0*](https://github.com/google-research-datasets/boolean-questions), and the [*MIT License*](https://github.com/rudinger/winogender-schemas/tree/master). In more restrictive cases, the dataset is public, but users must agree to contractual **[Terms of Service](glossary.html#terms-of-service)** to access it --- and those terms impose restrictions on how the dataset can be used. Finally, many datasets are public but cost money to access. For example, the [*UPenn Linguistics Data Consortium*](https://www.ldc.upenn.edu/) has a catalog of hundreds of high-quality datasets, but individuals need to be affiliated with a member institution of the consortium to access them.

## Open Model {#open-model}

The machine-learning community has described a **[model](glossary.html#model)** as **open-source** when a trained **[checkpoint](glossary.html#checkpoint)** has been released with a **[license](glossary.html#license)** allowing anyone to download and use it, and the software package needed to load the checkpoint and perform **[inference](glossary.html#inference)** with it have also been open-sourced. A model can be open-sourced even in cases where details about how the model was developed have not been made public (sometimes, such models are referred to instead as **semi-closed**). For example, the model creators may not have open-sourced the software package for training the model --- indeed, they may have not even publicly documented the training procedure in a technical report. Furthermore, a model being open-source does not necessarily mean the training **[data](glossary.html#datasets)** has been made public.

However, various pre-existing open communities (including the Open Source Initiative, which maintains the canonical Open Source Definition) have objected to usage by the machine-learning community, arguing that it does not capture several of the most important qualities of openness as it has been understood by the software community for over two decades. These qualities include the freedom to inspect the software, the freedom to use the software for any purpose, the ability to modify the software, and the freedom to distribute modifications to others.

## Open Software {#open-software}

**Open-source software** is software with source code that anyone can inspect, modify, and enhance, for any purpose. Typically, such software is licensed under a standardized open-source **[license](glossary.html#license)**, such as the [*MIT License*](https://opensource.org/license/mit/) or the [*Apache license*](https://opensource.org/license/apache-2-0/). Machine-learning systems typically consist of several relatively independent pieces of software; there is the software that builds the **[training dataset](glossary.html#datasets)**, the software that **[trains](glossary.html#training)** the model, and the software that does **[inference](glossary.html#inference)** with the **[model](glossary.html#model)**. Each of these can be independently open-sourced.

# Legal Concepts in Intellectual Property and Software {#gloss:legal-concepts}

## Claims {#claims}

**[Patent](glossary.html#patent)** **claims** are extremely precise statements that define the scope of protection within the patent. Patent claims are carefully written to be broad enough to encompass potential variations of the invention and specific enough to distinguish the invention from prior art.

## Copyright {#copyright}

**Copyright** grants exclusive rights to creators of original works. For a work to be copyrightable, it must meet certain criteria: (1) it must be original, and (2) it must possess a sufficient degree of creativity. Copyright does not protect facts or concepts, but expressions of those ideas fixed in a tangible medium (e.g., the idea for a movie, if not written down or recorded in some way, is typically not copyrightable; a screenplay is typically copyrightable). Copyright laws provide protections for various forms of creative expression, including, but not limited to, literary works, artistic works, musical composition, movies, and software [@copyright].

See **[idea vs. expression](glossary.html#idea-vs.-expression)**.

## Copyright Infringement {#copyright-infringement}

**Copyright infringement** occurs when someone uses, reproduces, distributes, performs, or displays copyrighted materials without permission from the copyright owner. This act breaches the exclusive rights held by the copyright holder.

## Damages {#damages}

In the context of **[IP](glossary.html#ip)**, **damages** refers to financial compensation awarded to the owner of IP for harms and losses sustained as a result of IP infringement. When IP rights such as **[patents](glossary.html#patent)**, **[copyright](glossary.html#copyright)**, or trademarks are violated, the owner of the IP may file a legal claim for damages. These can cover lost profits, reputational damage, or **[licensing](glossary.html#license)** fees.

## Fair Use {#fair-use}

**Fair use** is a legal concept that allows limited use of **[copyrighted](glossary.html#copyright)** materials without permission from the copyright owner [@fairuse]. Typically, fair use applies to contexts such as teaching, research, and news reporting, and fair use analyses consider the purpose of use, scope, and the amount of material used.

## The Field of Intellectual Property (IP) {#ip}

The field of **Intellectual Property (IP)** refers to a set of laws that grant exclusive rights for creative and inventive works. IP laws protect and promote ideas by providing incentives for innovation and protecting owners of inventions (e.g. written works, music, designs, among others). Intellectual property laws include **[copyright](glossary.html#copyright)**, **[patents](glossary.html#patent)**, trademarks and trade dress, and trade secrets. Intellectual property law is often the first recourse for conflicts around emerging technologies where more specific legislation has not yet crystallized. Property law is well-developed, widely applicable, and carries significant penalties including fines and forced removal or destruction of work.

## Harm {#harm}

Many areas of law only regulate actions that cause some identifiable **harm** to specific victims. For example, people who have not suffered individual harms may not have "standing" to bring a lawsuit in federal court. For another, **[damage](glossary.html#damages)** awards and other **remedies** may be limited to the harms suffered by the plaintiffs, rather than dealing more broadly with the consequences of the defendant's conduct. It is important to note that what counts as a cognizable harm is a legal question, and does not always correspond to people's intuitive senses of when someone has suffered a harm. Physical injury is the most obvious and widely accepted type of legal harm; other widely recognized forms of harm include damage to property, economic losses, loss of liberty, restrictions on speech, and some kinds of **[privacy violations](glossary.html#privacy-violation)**. But other cases have held that fear of future injury is not a present harm. Thus, having one's personal information included in a data breach may not be a harm by itself --- but out-of-pocket costs and hassle to cancel credit cards are recognized harms.

## Idea vs. Expression {#idea-vs.-expression}

This **idea vs. expression** dichotomy gets at the distinction between underlying concepts (or ideas) conveyed by a work, and the specific, tangible manner in which those are expressed. An idea refers to an abstract concept or notion behind a creative work, and ideas are not subject to **[copyright](glossary.html#copyright)** protection. However, expressions, as tangible manifestations, are. Tangible fixed expressions of ideas include words, music, code, or art. It is important to note that within copyright law, rights are granted to expression of ideas, not ideas themselves.

See **[copyright](glossary.html#copyright)**.

## License {#license}

A **license** gives legal permission or authorization, granted by the rights holder to others. License agreements explicitly outline rights that are granted, as well as limitations, restrictions, and other provisions related to its scope, for example, duration. Licenses are common practice within the **[field of IP](glossary.html#ip)**, and are commonly used in software, music, and film industries.

## Non-Expressive or Non-Consumptive {#non-expressive-or-non-consumptive}

Certain uses of **[copyrighted](glossary.html#copyright)** materials can be **non-expressive** or **non-consumptive**. In such cases, copyrighted material is used in a way that does not involve expressing or displaying original work to users. Some examples include text mining, building a search engine, or various forms of computational analyses.

## Patent {#patent}

A **patent** confers exclusive rights to inventors, granting them the authority to prevent others from making, using, or selling their inventions without permission. Patents create incentives for innovation by providing inventors with a time-based protection from the filing date. To obtain a patent, inventions must be new, inventive, and industrially applicable. Creators apply for patents; their applications must contain **[claims](glossary.html#claims)** that describe what is novel in the work.

## Prior Art {#prior-art}

**Prior art** is evidence of existing knowledge or information that is publicly available before a certain date. Prior art is critical in adjudicating the novelty and nonobviousness of a new invention and may include other **[patents](glossary.html#patent)**. Patent examiners search for prior art to determine the patentability of the **[claimed](glossary.html#claims)** invention. Further, prior art informs the patent's applicability and scope.

## Terms of Service {#terms-of-service}

**Terms of service (ToS)** refers to a contractual agreement between the **[IP](glossary.html#ip)** owner and users of **[licenses](glossary.html#license)** that govern the use and access to the protected content. ToS outline rights, restrictions, and obligations involved. ToS may specify permitted uses, licensing terms, and how IP may be copied or distributed. ToS safeguard IP owners' rights and ensure compliance with legal standards in the use of IP.

## Transformative Use {#transformative-use}

Expression can build on prior expression. In some cases, a new piece of **[copyrightable](glossary.html#copyright)** material may borrow or re-purpose material from prior work. If this new material creates something inventive, new, and substantially different from the original work, then it can be considered **transformative use** of the original work, as opposed to **[infringing](glossary.html#copyright-infringement)** on the original copyright owner's exclusive rights. The new material may also be copyright eligible. Parody is one common type of transformative use.

# Privacy {#app:privacy}

## Anonymization {#anonymization}

**Anonymization** is the process of removing or modifying personal data in a way that it cannot be attributed to an identifiable individual.

## The California Consumer Privacy Act (CCPA) {#ccpa}

The **CCPA** is a California state law that provides consumers with the right to know what personal information businesses collect about them, the right to request their personal information be deleted, and the right to opt-out of sales of their personal information. The CCPA applies to all businesses that operate in California, as well as those outside of California that may transfer or process the personal information of California residents.

## Consent {#consent}

**Consent** is the voluntary and informed agreement given by an individual for the collection, use, or disclosure of their personal information. In the context of data, consent often requires clear and specific communication about the purpose and use of collected data.

## Differential Privacy {#differential-privacy}

**Differential privacy** (DP) is an approach for modifying algorithms to protect the membership of a given **record** in a dataset [@dp]. Informally, these guarantees are conferred by adding small amounts of **noise** to the individual data **[examples](glossary.html#examples)** in the dataset. Let us say that there are two version of a dataset, $D$ and $D'$, where the former contains an example $E$ and the latter does not. If we were to run differentially private **[algorithms](glossary.html#algorithm)** to compute statistics on the datasets $D$ and $D'$, we would not be able to tell by those statistics which dataset contains $E$ and which does not. As a result, we can no longer use the computed statistics to infer whether or not the original training data contained the example $E$.

Differential privacy is a theoretical framework that encounters some challenges in practice. For example, the amount of noise one must add to data may impact the accuracy of statistics computed, or, when used for generative-AI models, may impact performance of the model. On the other hand, someone using a differentally private approach needs to add enough noise to ensure that the two datasets $D$ and $D'$ cannot be differentiated through computed statistics. Finally, differential privacy was originally created for tabular data and encounters challenges adapting to the unstructured data commonly used for generative-AI models. For more on the challenges of applying differential privacy to **[large language models](glossary.html#llm)**, please see @brown2022privacy.

## The General Data Protection Regulation (GDPR) {#gdpr}

**GDPR** is a comprehensive data protection law implemented by the European Union in 2018 [@gdpr]. The GDPR governs the collection, use, storage, and protection of personal data for EU residents. The law sets out specific rights for individuals regarding their personal data, such as the right to access, rectify, and delete their data, as well as the right to know how their data is being processed. Further, the GDPR imposes obligations on organizations such as businesses that handle personal data to ensure that proper data protection measures are in place and that **[consent](glossary.html#consent)** is obtained for data processing. Non-compliance results in fines and penalties.

## Personally Identifiable Information (PII) {#pii}

**Personally Identifiable Information (PII)** refers to data that can be used to identify an individual. PII can include names, addresses, phone numbers, social security numbers, email addresses, financial information, and biometric data. PII is sensitive, and organizations that collect PII are required to implement appropriate measures, adhering to relevant data protection laws (such as the **[GDPR](glossary.html#gdpr)**) to safeguard its confidentiality and integrity.

## Privacy Policy {#privacy-policy}

A **privacy policy** consists of documents that outline how organizations collect, use, store, and protect personal information. Privacy policies are meant to inform individuals about their rights and the organization's data processing practices.

## Privacy Violation {#privacy-violation}

A **privacy violation** involves unauthorized or inappropriate intrusion into an individual's personal information or activities. Privacy violations may occur in various forms from data breaches, surveillance, identity theft, or sharing personal or sensitive information without **[consent](glossary.html#consent)**. These violations may lead to significant harm such as the loss of personal autonomy, reputational damage, or financial loss.

## The Right to be Forgotten {#the-right-to-be-forgotten}

Some countries' legal systems recognize a **right to be forgotten** that grants individuals the ability to request the removal of their personal information from online platforms or search-engine results. The idea is that the legitimate public interest in knowing about other people's past conduct can be outweighed when the information about it is out-of-date or misleading. The European Union's **[GDPR](glossary.html#gdpr)** includes a form of the right to be forgotten.

## Tort {#tort}

A **tort** is a civil wrongdoing that causes **[harm](glossary.html#harm)** or injury to another person or their property. Tort law provides remedies and compensation to individuals who suffer harm as a result of someone else's actions or negligence.

# References