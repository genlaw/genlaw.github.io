---
title: "Report of the 1st Workshop on Generative AI and Law [blog]"
tags: [machine learning, datasets, generative ai, artificial intelligence, copyright, privacy, law]
return-genlaw: true
return-footer: true

authors: 
  - name: A. Feder Cooper\*, Katherine Lee\*, James Grimmelmann\*, Daphne Ippolito\* and 31 other authors across 25+ institutions
  - affil: "*Equal contribution. All of the listed authors on the report contributed to the workshop upon which the report and this associated blog post  are based, but they and their organizations do not necessarily endorse all of the specific claims in this report. Correspondence: genlaw.org@gmail.com. We thank Matthew Jagielski, Andreas Terzis, and Jonathan Zittrain for feedback on this report."


readon:
  arxiv: https://arxiv.org/abs/2311.06477
  pdf: https://genlaw.github.io/2023-report.pdf
  ssrn: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4634513
---


The [Report of the 1st Workshop on Generative AI and Law](2023-full-report.html) reflects the synthesis of ideas from our Day 2 roundtables.
The report begins with a brief framing note about the impact of [Generative AI on Law](#section2), then goes on to suggest useful components of a [shared knowledge base](#section3), an outline of ways that [Generative AI is unique](#section4) (and the ways it isn’t), a preliminary [taxonomy](#section5) of legal issues, and a concrete [research agenda](#section6) at the intersection of Generative AI and law.  
The report closes with some [brief takeaways](#section7) about this emerging field. 

# The Impact of Generative AI on Law {#section2}
*[Section 2](2023-full-report.html#sec:vision)*

We begin the report with some background that helps situate why Generative AI is going to have such an impact on law. 
It’s true that  Generative AI is “generative” because it generates text, images, audio, or other types of output. 
But it is also “generative” in the sense of Jonathan Zittrain’s theory of generative technologies from 2008: it has the “capacity to produce unanticipated change through unfiltered contributions from broad and varied audiences”[ @zittrainfuture].
It provides enormous leverage across a wide range of tasks, is readily built on by a huge range of users, and facilitates rapid iterative improvement as those users share their innovations with each other.
As a result, generative-AI systems will be both immensely societally significant — too significant for governments to ignore or to delay dealing with — and present an immensely broad range of legal issues.






# Developing a Shared Knowledge Base {#section3}
*[Section 3](2023-full-report.html#sec:knowledge)*

## Crafting Glossaries and Metaphors

At GenLaw, it quickly became apparent that the two communities may share words, but these words may not share meanings. 
We therefore developed a [glossary](glossary.html) and list of [metaphors](metaphors.html) to make definitions of terms concrete, so that we can be sure that we’re talking about the same things (and talking about them precisely) 
Drawing from the [report](2023-full-report.html), we use “harm” as an example of an overloaded term — on that has a colloquial understanding that can be mistaken for a term-of-art in the law: 


> 
> Many technologists were not aware of the importance of [harms](glossary.html#harm) as a specific and consequential concept in law, rather than a general, non-specific notion of unfavorable outcomes.
> We found our way to common understandings only over the course of our conversations, and often only after many false starts.


We also suggest that [metaphors](metaphors.html) can be a useful abstraction for communication between communities,  since they play a central role to how both machine-learning experts and lawyers communicate among themselves. 
Lawyers use metaphors as rational frameworks for thinking through the relevant similarities and differences between cases. 
 In the machine-learning community, experts use metaphors all the time to give intuitions for technical processes.
For example, generative models are said to “[learn](metaphors.html#learning)” or “[make collages](metaphors.html#collage)” of training data. 
This imaginative naming is often intentional; technical processes are often  named for the human behaviors or science-fiction tropes that inspired them. 

However, we caution that metaphors can also simplify and distort (as is the case with the metaphor of a [collage](metaphors.html#collage)). 
For better communication across fields, it can nevertheless be instructive to understand the ways a metaphor is appropriate and where it falls short. 



## *Understanding Evolving Business Models*
*[Section 3.3](2023-full-report.html#sec:business)*

Generative AI is not a single entity or business model. 
There are many different types of Generative AI built by a diversity of actors, potentially in partnership. 
To get a better understanding of the array of generative-AI systems and the ways that they’re produced, we can look at existing and emerging business models: 1)  business-to-consumer (B2C) hosted services  (including direct to consumer applications, e.g., OpenAI’s ChatGPT, Google’s Bard) and [application programming interfaces (APIs)](glossary.html#api)); 2)  business-to-business (B2B) integration with hosted services ( via direct partnership/integration or through the use of APIs),  3) products derived from [open-source](glossary.html#app:os) software, models, and datasets (e.g., ,some versions of Stable Diffusion, offered by Stability AI, are open sourced), and 4) companies that operate at specific points in the generative-AI [supply chain](glossary.html#supply-chain) [@lee2023talkin]  (e.g., companies that work specifically on [datasets](glossary.html#datasets), training diagnostics, and [training](glossary.html#training) and deployment). We go into more detail on each of these in the report.


# Pinpointing Unique Aspects of GenAI {#section4}
*[Section 4](2023-full-report.html#sec:challenges)*

With so many different types of generative-AI systems, some of the lawyers in the room asked the ML experts to clarify some commonalities that make the “magic” of Generative AI. 
We discussed three aspects:

1. **Open-ended tasks:** Generative AI models are trained with open-ended tasks in mind, rather than  narrowly-defined tasks. 
This means that the same model could be used for translating between languages as could be used for question answering. 
2. **Multi-stage pipelines:** In part as a result of training with open-ended tasks, models are trained in a _multi-stage training pipeline_ containing stages like: [pre-training, fine-tuning](glossary.html#pre-training-and-fine-tuning), and [alignment](glossary.html#alignment) (e.g., [RLHF](glossary.html#reinforcement-learning)). 
The delineations between these different stages is flexible, but the result is to create [base models](glossary.html#base-model) that have a “base” of knowledge about the world within the model. 
This training pipeline is part of a larger supply chain, which further contributes to novel dynamics in the  production and use of generative-AI systems [@lee2023talkin]. 
3. **Scale:** Finally, arguably the most discussed element of Generative AI was the role of _scale_: _scale_ of datasets, of models, of the number of generations, and of compute.

> [Pre-training](glossary.html#pre-training-and-fine-tuning)
> : One of the _ah-ha_ moments we had at the workshop was when we realized that technologists and legal scholars understood the term _pre-training_ to mean very different things.
> Technologists use the term pre-training to refer to an early, general-purpose phase of the model training process, but legal scholars assumed that the term referred to a data preparation stage prior to and independent of training. 
> Clearing up that confusion made the importance of pre-training models much more apparent.


# A Preliminary Taxonomy of Legal Issues {#section5}
*[Section 5](2023-full-report.html#sec:taxonomy)*

We also outlined some of the legal issues Generative AI raises. ^[We focus on the workshop’s intended scope of [privacy](glossary.html#app:privacy) and [intellectual property (IP)](glossary.html#ip) issues. This is by no means a comprehensive taxonomy of legal issues nor of harms (legally cognizable or otherwise). Other reports have made significant attempts to catalog such harms from Generative AI, for example @epic.]
Not all issues that Generative AI raises are _new_. 
Generative AI can be used to perform many tasks for which other AI/ML technology has already been commonly used.^[For example, instead of using a purpose-built sentiment-analysis model, one might simply prompt an LLM with labeled examples of text and ask it to classify text of interest; one could use a trained LLM to answer questions with “yes” or “no” answers (i.e., to perform classification).]
In the report, we focus on  four areas where Generative AI raises novel challenges for the law: intent, privacy, misinformation and disinformation, and intellectual property.



* **Intent:** Generative AI can cause harms that are similar to those brought about by human actors, but do so without human intention.
* **Privacy:** Generative AI presents new privacy challenges and complicates existing ones. Models are trained on large-scale datasets that may contain all sorts of private information (e.g., [PII](glossary.html#pii)), which in turn can be memorized and then leaked in generations [@brown2022privacy; @carlini2023extracting]. 
* **Misinformation and Disinformation:** Contending with misinformation and disinformation online is not a new challenge. However, easy, cheap generation disinformation about individuals (e.g., deepfakes) could contribute to new types of  harms.
* **Intellectual Property:** The copyright issues raised by generative-AI systems are all over the news.^[Much has been written and said on copyright and Generative AI, e.g.,  @lee2023talkin, @sag2023safety, @samuelson, @CallisonBurch_2023, @vyas2023provable, @lipton2023privacy.]
But, Generative AI raises other issues in [intellectual property](glossary.html#ip) more broadly. 
For example, it  raises numerous questions  around patents,^[Regarding inventorship, how should ownership rights for a drug designed using generative-AI tools be allocated?]  volition (see below), market externalities, trade secrecy,  scraping, and more. 

For more detail on each of these, please see the [report](2023-report.html#sec:taxonomy).


> Volition
> : Human volition plays an important and subtle role in defining IP infringement.
> For example, copyright infringement normally requires that a human intentionally made a copy of a protected work, but not that the human was consciously aware that they were infringing. Generative-AI systems may occasionally produce outputs that look like duplicates of the training data. 
> Some participants at GenLaw were concerned that it may be easy to deflect the role of human-made design choices by making such choices seem “internal” to the system (when, in fact, such choices are typically not foregone conclusions or strict technical requirements).

# Toward a Long-Term Research Agenda {#section6}
*[Section 6](2023-full-report.html#sec:agenda)*

Through our discussions, we elicited several important and promising research directions.
Each of these directions brings forth challenges that require engagement from law and machine-learning experts, and likely many other disciplines as well.

1. **Centralization versus Decentralization:** First, who will build the components of future generative-AI systems? How centralized or decentralized will these actors be? This is as much a technical question as it is a business and legal question. Technical constraints, such as the design of a dataset, inform the logistics of the supply chain: who builds the component, what gets built, and how. Improvements in synthetic data may enable well-resourced actors to generate their own training data. Existing and emerging business models create incentives for particular modes of interaction among players. Finally, every important potential bottleneck in Generative AI – from datasets to compute to models and beyond – will be the focus of close scrutiny. These questions cannot be discussed intelligently without contributions from both technical and legal scholars.

2. **Rules, Standards, Reasonableness, and Best Practices**: In some cases, we have standards of care. For example, HIPAA strictly regulates which kinds of data are treated as personally identifying and subject to stringent security standards. 
In other cases, we rely on _reasonable_ standards of practice; but, what is _reasonable_ is often both context-dependent and evolving..
What is _reasonable_ will depend on technical advancements and constraints.
This is an area where we feel that collaboration between legal and technical experts  and policymakers is urgently needed. 

3. **Notice and Takedown ≠ Machine Unlearning:** Notice and takedown requests are particularly challenging for generative-AI models.
The impact of each  example in the training data is dispersed throughout the model once it is trained and cannot be easily traced. 
There are entire subfields of machine learning devoted to problems like these, such as machine unlearning and attribution. However, both machine unlearning and attribution are very young fields, and their strategies are (for the most part) not yet computationally feasible to implement in practice for deployed generative-AI systems. There is, of course, intense (and growing) investment in this area. 

4. **Evaluation Metrics:** Effective ways to evaluate generative-AI systems currently remain elusive. System capabilities and harms are not readily quantifiable; designing useful metrics will be an important, related area of research for Generative AI and law (and will also, in turn, influence what we understand to be _reasonable_ system behaviors). 

# What’s next for GenLaw? {#section7}
As is clear from the diversity of issues discussed within these topics, it is difficult to pithily sum up the main takeaways of GenLaw. (Nevertheless, we attempt to do so in the [report](2023-report.html).) 

To close here, we just want to say that we’re so thrilled to have been able to have hosted the first GenLaw, host this discussion and for the community that’s grown up around it. 
We hope that you refer to and share our report, [glossary](glossary.html), [metaphors](metaphors.html), [resources](resources.html), [law review article](talkin.html), and [explainers series](explainers.html) (written for and with workshop participants)  as reference material for your own learning, teaching materials, and research.

Right now, we’re growing GenLaw into a nonprofit, which will be a home for research, education, and interdisciplinary discussion. 
We will continue to create resources for both a general audience and subject-matter experts. 
So far, we’ve brought together experts across Generative AI, law, policy, and other
computer-science disciplines from 25 different institutions, and we are excited to continue engaging with experts across industry, academia, and government. 
While our first event and materials have had a U.S.- based orientation, we are actively focusing on expanding our engagement globally. 


Stay tuned for more from us. 
You can subscribe to updates [here](https://groups.google.com/g/genlaw-community). 

And, of course, a big thank you to our sponsors: Google, Microsoft, Schmidt Futures, OpenAI, Anthropic, Cornell Law School, and ML Collective.


# References


