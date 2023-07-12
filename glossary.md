---
title: "GenLaw: Glossary"
tags: [machine learning, datasets, generative ai, artificial intelligence, copyright, privacy, law]
return-genlaw: true
---
<p style="text-align:right">Also see the [resources](resources.html).</p>

# Machine Learning and Artificial Intelligence
## Core Concepts in Machine Learning
### Model

The core of contemporary machine learning is the idea of a **model**. A model is a mathematical tool that takes an **input** and produces an **output**. A simple example might be a model that tells you whether a temperature is above or below average for a specific location. In this case the input is a number (temperature) and the output is binary (above/below). 

There could be many versions of this model depending on the location. The behavior of the model is therefore defined by an internal **parameter** (or, **weight**). In our temperature example, the model has one parameter, the average temperature for the location. The process for setting the value of the parameter for a specific version of the model is called **training**. In this case we might train a model for New York City by gathering historical temperature data and calculating the average. The process of gathering historical data is called **data collection**. The process of training — in this case, calculating the average — is an **algorithm.** 

A saved copy of a model's trained parameters is called a **checkpoint**. We might save separate checkpoints for different cities or save new checkpoints for our NYC model if we retrain with new data. The process of applying the model to new inputs is called **inference**. To create an output for a temperature, we also apply an algorithm: subtract the parameter from the input, and return "above" if the difference is positive. A common type of model **architecture** is a **neural network.**

### Parameter / weight
**Parameters** are numbers that define the specific behavior of a model. For example, a linear regression model that predicts the number of bicycle trips given temperature and rainfall might have three parameters: one representing the effect of temperature, one representing the effect of rainfall, and one additional **intercept** or **bias** parameter representing the baseline number of trips. Contemporary **neural network** models have millions to billions of parameters. The values of parameters are saved in files called **checkpoints**.

### Datasets and examples
Machine learning **models** are usually defined as functions with optimizable **parameters** that produce an output for a given input. In order to find settings for numerical parameters, we need **examples** of input-output pairs. A collection of such input-output pairs is a **dataset**. 

Datasets are often copied and reused for multiple projects because (a) they are expensive and time-consuming to produce and (b) reusing datasets makes it easier to compare new models and algorithms to existing methods (a process often called **benchmarking**). It is generally assumed that the individual examples in a dataset are not important except as representative **samples** from a **probability distribution** of input-output pairs, which  the model is likely to encounter when in  use. As such, **memorization** of specific input-output pairs is considered a failure, while **generalization** to the distribution (i.e., a trained model performing well on previously unseen examples from the same distribution) is considered success. 

Datasets are often divided into **training** and **testing** portions to distinguish memorization from generalization. Commonly used datasets and sources include: 

* [Wikipedia](https://huggingface.co/datasets/wikipedia)
* [Project Gutenberg](https://arxiv.org/abs/1812.08092v1)
* [Common Crawl](https://commoncrawl.org/)
* [The Pile](https://pile.eleuther.ai/)
* [C4](https://github.com/allenai/allennlp/discussions/5056)
* [ImageNet](https://www.image-net.org/)
* [ROOTS](https://huggingface.co/papers/2303.03915)

### Neural network
A neural network is a way of using simple, easy-to-learn relationships to model complex, difficult relationships. For example, if you can only draw straight lines, it is impossible to draw a circle. But if you _combine_ many straight lines you can draw something that looks like a curve. The model synthesizes multiple simple decisions by passing the input through a series of intermediate transformations. Many networks consist of **layers**, where the output of one layer is used as the input of the next layer. Each layer consists of a set of classifiers (**neurons**) that each performs a simple operation independently of one another. The outputs of all classifiers at layer _n_ are then passed to each classifier in layer _n+1_, and so forth. Each classifier in each layer has **parameters** that define how it responds to input.

### Architecture
The design of a model is called its **architecture**. For a **neural network,** the design process requires making several decisions, e.g., the size of inputs that can be accepted, the number of layers, how many **parameters** per layer, how they are connected, and how we represent the intermediate state of the model as each layer transforms input to output. 

Many contemporary models appear in **families** that have similar architectures but differing scales, often referred to by the total number of parameters. For example, Meta released four sizes of the LLAMA family that differ only in the number of layers and size of intermediate **vector representations**. More layers and wider internal representations can improve the capability of a model, but can also increase the amount of time it takes to **train** the model or to do **inference.** 


### Algorithm
An **algorithm** is a formal specification of a process. Machine learning uses algorithms for **training** models and for operating models on new data, e.g., for **inference.** In public discourse around social media, algorithm has come to refer to methods for **optimizing** the probability that a user will engage with a post; however, it is important to note that the process of simply sorting posts by date also requires an algorithm.

### Training
The process of setting the value of numeric **parameters** of a **model** based on **data** is called **training**. A common **algorithm** for training neural network models is **stochastic gradient descent,** or SGD. Training data sets are often too large to process all at once, so SGD operates on small **batches** of dozens to hundreds of **examples** at a time. For each example, the algorithm generates the model's output based on the current setting of the parameters and compares that output to the desired output from the training data (e.g. did we correctly choose the next word?). If the output did not match, the algorithm works backwards through the model's layers and modifies parameters so that the correct output becomes more likely. This process can be thought of as leaving "echoes" of the training examples encoded in the parameters of the model.

### Hyperparameter
**Parameters** are numbers in a model whose values are set during training. There are additional numbers that are _not_ learned that the model builder sets prior to training, which are often manually specified. Some of these are properties of the architecture, such as input length, internal dimension, and number of layers. Others determine the behavior of the training algorithm, such as the learning rate, which controls how much we update model parameters after each input/output training example. Such parameters are called **hyperparameters,** and they can have a big effect on the overall behavior of trained models. 

## Concepts in Generative AI
### Language model
A language model (LM) is a type of **model** that takes a sequence of text as input and returns a numeric score as output. Many language models are designed to divide input into a **prefix** and a **continuation**, where only the continuation is scored. A language model can be used to implement many tasks, such as autocomplete (given your current text, what are the three most likely next words?) or translation (which is more likely, "recognize speech" or "wreck a nice beach"?). Systems like ChatGPT take the user's input text as a prefix and then use the scores from a language model to generate a response one word at a time: After each word is selected, the system appends that word to the previous prefix to create a new prefix, which is then used to score candidates for the next word. Different types of language models have been studied for more than 100 years. Contemporary LMs are usually **neural networks**.

### Large language model
This term has become popular as a way to refer to the current spate of language models that use the **Transformer architecture**. It does not have a specific technical definition, but usually implies a number of **parameters** on the order of hundreds of million or more (a number that has continually evolved).

### Diffusion model
A **diffusion model** is the model architecture behind models like Stable Diffusion. Diffusion models are commonly used for images and are trained to transform a noisy image into a target image (a given image from the training set). The noisy image is so noisy that it starts out as pure noise, and the training process will also reward the model for exactly reconstructing the target image.



### Transformer
A **transformer** is a popular **model architecture** for image, text, and music applications and underlies models like ChatGPT, Bard, and MusicLM. An input (text or image) is broken into segments (word **tokens** or image patches) as a preprocessing step. These input segments are then passed through a series of layers that generate **vector representations** of the segments. What makes the transformer model distinct is that the strength of connection between segments is itself a learned behavior. The model has trainable parameters that determine how much **attention** to parts of the input.  Like the diffusion model, the transformer model is rewarded for reproducing a target training example exactly.

### Tokenization
For language models, a common preprocessing step is to break documents into segments called **tokens**. For example, the input "I like ice cream." might be tokenized into ["I", "like", "ice", "cream", "."]. The tokens can then be mapped to entries in a **vocabulary** of strings that are recognized by a model. Each model family tends to share a vocabulary, which is optimized to represent a particular training corpus. Most current models use **subword tokenization** to handle words that would otherwise not be recognized. Therefore, a rare or misspelled word might be represented by multiple tokens. The number of tokens used to represent an input is important because it determines how large the effective input size of a model is.

### Pre-training, Fine-tuning, and Foundation Models
Current protocols divide **training** into a common **pre-training** phase that results in a general-purpose or **foundation** model and an application-specific **fine-tuning** phase that adapts a pretrained model **checkpoint** to perform a desired task using additional data. This paradigm has become common over the last five years, especially as model architectures have become larger and larger. This is because, relative to pretraining, finetuning is fast and inexpensive: It is much cheaper to fine-tune an existing foundation model for a particular task than it is to always train a new model from scratch. 

Because pretrained models are most useful if they provide a good basis for many distinct applications, model builders have a strong incentive to collect as much pre-training data from as many distinct sources as possible. Fine-tuning results in a completely new model checkpoint (potentially gigabytes of data that must be loaded and served separately from the original model), and requires hundreds to thousands of application-specific examples. 

### Multimodal
Computers represent different ***modalities*** of input data in distinct ways, including text, images, and audio. Multimodal models are able to process inputs in multiple forms, for example by mapping both images and documents into a shared **embedding** representation or by generating a text caption from an image.

### Generation
**Generative models** produce complex, human interpretable outputs such as full sentences or natural-looking images, called **generations.** More traditional , machine learning models are limited to ranges of numeric outputs (**regression**) or discrete output labels like “cat” and “dog” (**classification**). The process of producing generations is much more difficult: The vast majority of possible images just look like static, and even high-quality generations can reach an **uncanny valley** with subtly wrong details like seven-fingered hands. Note that in the 2000s, "generative" was used to describe a method for designing ML algorithms using Bayes rule; this is not the same meaning. 

### Hallucination
Generative models are trained to produce outputs based on training data. But they don’t have explicit representations of facts or knowledge and sometimes generate unrealistic content, such as images of the pope wearing a puffy white coat or a hedgehog driving a train, or "bibliographies" consisting of research papers that do not exist. It has become common to refer to such outputs as [**hallucinations**](https://www.wsj.com/articles/hallucination-when-chatbots-and-people-see-what-isnt-there-91c6c88b?st=wns4rqlp2dl1ly5&reflink=desktopwebshare_permalink).

### Reinforcement learning
Reinforcement learning is a method for incorporating feedback into systems. For some applications, solving a given  task requires finding an optimal policy or strategy, and feedback may only be available after many decisions have been made. For example, a chess system needs a function that decides between possible moves, but you may only find out whether a move was good many moves later when you either win or lose. 
RL is currently being used to improve generation with responses to  **human feedback** about whether the generations were “good” or “useful.”

### In-context learning / zero-shot / few-shot

In many cases we can use **pretrained models** directly without creating new **checkpoints** through **fine-tuning**. This mode of use is important because it supports lightweight API-based access through a common deployed model. **In-context learning** involves creating an input or **prompt** to a model in a way that defines a desired output. Typically the input includes either **instructions** (a natural language description of the output) or a small number of **examples** of input-output pairs. **Shot** refers to these examples: **zero-shot** provides only instructions, **two-shot** provides two input-output pairs along with optional instructions, and so forth.

### Instruction fine-tuning

**Fine-tuning** models to "follow instructions" has become an important special case with the popularity of ChatGPT. Examples of interactions in which someone makes a request and someone else follows those instructions are relatively rare on the internet compared to, for example, question / answer forums. As a result, such data sets are often constructed specifically for the purpose of language model finetuning, and may provide substantial practical benefits for commercial companies. Instruction finetuning can also take into account legal, ethical, or other normative constraints. Instruction finetuning can be critical in enabling **zero-shot** or **few-shot** use patterns.


# Legal Concepts

## Intellectual Property (IP)

### The Field of IP

A set of laws that grant exclusive rights for creative and inventive works. IP laws protect and promote ideas by incentivizing innovation and protecting owners of inventions (e.g. written works, music, designs among others). Intellectual property laws include copyright, patents, trademarks and trade dress, trade secrets. 

### Copyright

[**Copyright**](https://www.law.cornell.edu/uscode/text/17/102) grants exclusive rights to creators of original works. For a work to be copyrightable, it must meet a certain criteria: (1) it must be original, and (2) it must possess a sufficient degree of creativity. Copyright does not protect facts or concepts, but expressions of those ideas fixed in a tangible medium (e.g., the idea for a movie, if not written down or recorded in some way, is typically not copyrightable; a screenplay is typically copyrightable). Copyright laws provide protections for various forms of creative expression, including, but not limited to literary works, artistic works, musical composition, movies, and software. 

### Copyright Infringement

Copyright infringement occurs when someone uses, reproduces, distributes, performs, or displays copyrighted materials without permission from the copyright owner. This act breaches the exclusive rights held by the copyright holder.  

### Transformative Use

Expression can build on prior expression. In some cases, a new piece of copyrightable material may borrow or re-purpose material from prior work. If this new material creates something inventive, new, and substantially different from the original work, then it can be considered **transformative use** of the original work, as opposed to **infringing** on the original copyright owner’s exclusive rights. The new material may also be copyright eligible. Parody is one common type of transformative use.

### Fair Use
[**Fair use**](https://www.law.cornell.edu/uscode/text/17/107) is a legal concept that allows limited use of copyrighted materials without permission from the copyright owner. Typically, fair use applies to contexts such as teaching, research, and news reporting, and fair use analyses consider the purpose of use, scope, and the amount of material used.

### Non-Expressive or Non-Consumptive

Certain uses of copyrighted materials can be **non-expressive** or **non-consumptive**. In such cases, copyrighted material is used in a way that does not involve expressing or displaying original work to users. Some examples include text mining, building a search engine, or various forms of computational analyses.

### Patent
A **patent** confers exclusive rights to inventors, granting them the authority to prevent others from making, using, or selling their inventions without permission. Patents create incentives for innovation by providing inventors with a time-based protection  from the filing date. To obtain a patent, inventions must be new, inventive, and industrially applicable. Creators apply for patents; their applications must contain **claims** that describe what is novel in the work. 

### Claims

Patent **claims** are extremely precise statements that define the scope of protection within the patent. Patent claims are carefully written to be broad enough to encompass potential variations of the invention and specific enough to distinguish the invention from prior art.

### Prior Art

**Prior** art is evidence of existing knowledge or information that is publicly available before a certain date. Prior art is critical in adjudicating the novelty and nonobviousness of a new invention and may include other patents. Patent examiners search for prior art to determine the patentability of the claimed invention. Further, prior art informs the patents applicability and scope.

### Idea vs. Expression
This **idea vs. expression** dichotomy gets at the distinction between underlying concepts (or ideas) conveyed by a work, and the specific, tangible manner in which those are expressed. An idea refers to an abstract concept or notion behind a creative work, and ideas are not subject to copyright protection. However, expressions as tangible manifestations, are.  Tangible fixed expressions of ideas can be words, music, code, or art. 

It is important to note that within copyright law, rights are granted to expression of ideas, not ideas themselves (see **copyright** above). 

### License

Legal permission or authorization granted by the rights holder to others. **License** agreements explicitly outline rights that are granted, as well as limitations, restrictions, and other provisions related to its scope, for example, duration. Licenses are common practice within the field of IP, and are commonly used in software, music, and film industries.

### Terms of Service
**Terms of service (ToS)** refers to a contractual agreement between the IP owner and users of licenses that govern the use and access to the protected content. ToS outline rights, restrictions, and obligations involved. ToS may specify permitted uses, licensing terms, and how IP may be copied or distributed. ToS safeguard IP owners’ rights and ensure compliance with legal standards in the use of IP. 

### Damages
**Damages** refers to financial compensation awarded to the owner of IP for harms and losses sustained as a result of IP infringement. When IP rights such as patents, copyright, or trademarks are violated, the owner of the IP may file a legal claim for damages. These can cover lost profits, reputational damage, or licensing fees. 

## Privacy

### Privacy violation

A **privacy violation** involves unauthorized or inappropriate intrusion into an individual’s personal information or activities. Privacy violations may occur in various forms from data breaches, surveillance, identity theft, or sharing personal or sensitive information without consent. These violations may lead to significant harm such as the loss of personal autonomy, reputational damage or financial loss. 

### Consent

**Consent** is the voluntary and informed agreement given by an individual for the collection, use, or disclosure of their personal information. In the context of data, consent often requires clear and specific communication about the purpose and use of collected data.

### Privacy Policy

A **privacy policy** consists of documents that outline how organizations collect, use, store, and protect personal information. Privacy policies are meant to inform individuals about their rights and the organization’s data processing practices. 

### Anonymization

**Anonymization** is the process of removing or modifying personal data in a way that it cannot be attributed to an identifiable individual. 

### Right to be forgotten
**The right to be forgotten** grants individuals the ability to request the removal of their personal information from online platforms or search engine results. This principle is based on the notion that individuals should have control over their digital information and the ability to safeguard their privacy. 

### The General Data Protection Regulation (GDPR)
The [**GDPR**](https://gdpr-info.eu/) is a comprehensive data protection law implemented by the European Union in 2018. The GDPR governs the collection, use, storage, and protection of personal data for EU residents. The law sets out specific rights for individuals regarding their personal data such as the right to access, rectify, and delete their data, as well as the right to know how their data is being processed.

Further, the GDPR imposes obligations on organizations such as businesses that handle personal data to ensure that proper data protection measures are in place and that consent is obtained for data processing. Non-compliance results in fines and penalties.

### Tort
A **tort** is a civil wrongdoing that causes harm or injury to another person or their property. Tort law provides remedies and compensation to individuals who suffer harm as a result of someone else's actions or negligence. 

### Personally Identifiable Information (PII)
**Personally Identifiable Information (PII)** refers to data that can be used to identify an individual. PII can include names, addresses, phone numbers, social security numbers, email addresses, financial information, and biometric data. PII is sensitive, and organizations that collect PII are required to implement appropriate measures, adhering to relevant data protection laws (such as the GDPR) to safeguard its confidentiality and integrity.

### The California Consumer Privacy Act (CCPA)

The **CCPA** is a California state law that provides consumers with the right to know what personal information businesses collect about them, the right to request their personal information be deleted, and the right to opt-out of sales of their personal information. The CCPA applies to all businesses that operate in California, as well as those outside of California that may transfer or process the personal information of California residents.


<p style="text-align:right">Jun 1, 2023</p>
<p style="text-align:right">Written by: [A. Feder Cooper](https://afedercooper.info/), [David Mimno](https://mimno.infosci.cornell.edu), [Madiha Z. Choksi](https://madihaz.com/), and [Katherine Lee](https://katelee168.github.io)</p>