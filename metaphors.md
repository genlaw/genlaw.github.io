--- 
title: "GenLaw: Metaphors" 
tags: [machine learning, datasets, generative ai, artificial intelligence, copyright, privacy, law] 
return-genlaw: true 
return-footer: true
--- 

We briefly discuss several metaphors for Generative AI that came up in the GenLaw discussions. It is worth considering why these metaphors are helpful and where they start to break down.

See also: [resources](resources.html), [glossary](#glossary.html), and the [GenLaw Report](2023-report.html)

## Models are trained. {#training}

Machine-learning practitioners will often say they **[train](glossary.html#training)** **[models](glossary.html#model)**. Training brings to mind teaching a dog to perform tricks by enforcing good behavior with treats. Each time the dog performs the desired behavior, they get a treat. As the dog masters one skill it may move onto another. Model training is similar in the sense that models are optimized to maximize some **[reward](glossary.html#reward)**.[^1] This reward is computed based on how similar the model's outputs are to desired outputs from the model.

However, unlike training a dog, model training does not typically have a curriculum;[^2] there is no progression of easier to harder skills to learn, and the formula for computing the reward remains the same throughout model training.

## Models learn like children do. {#learning}

**Learning** is the active verb we use to describe what a **[model](glossary.html#model)** does as it is being **[trained](glossary.html#training)** --- a model is *trained*, and during this process it *learns*. Model learning is the most common anthropomorphic metaphor applied to machine-learning models. The use of the word **learning** by machine-learning practitioners has naturally led to comparisons between how models learn and how human children do. Both children and machine-learning models are "skilled imitators," acquiring knowledge of the world by learning to imitate provided exemplars. However, human children and Generative AI obviously use very different mechanisms to learn. Techniques that help generative-AI systems to learn better, such as increasing model size, have no parallels in child development; mechanisms children use to "extract novel and abstract structures from the environment beyond statistical patterns" have no machine-learning comparisons [@yiu2023imitation].

## Generations are collages. {#collage}

We quote directly from discussion in @lee2023talkin [p. 58], with added links to our glossary.

> It also may seem intuitively attractive to consider **[generations](glossary.html#generation)** to be analogous to collages. However, while this may seem like a useful metaphor, it can be misleading in several ways. For one, an artist may make a collage by taking several works and splicing them together to form another work. In this sense, a generation is not a collage: a generative-AI system does not take several works and splice them together. Instead, ... generative-AI systems are built with **[models](glossary.html#model)** **[trained](glossary.html#training)** on many **[data examples](glossary.html#examples)**. Moreover, those data examples are not explicitly referred back to during the generation process. Instead, the extent that a generation resembles specific data examples is dependent on the model **[encoding](glossary.html#vector-representation)** in its **[parameters](glossary.html#parameters)** what the specific data examples look like, and then effectively recreating them. Ultimately, it is nevertheless possible for a generation to look like a collage of several different data examples; however, it is debatable whether the the process that produced this appearance meets the definition for a collage. There is no author "select[ing], coordinat[ing], or arrang[ing]"[^3] training examples to produce the resulting generation.

## Large language models are stochastic parrots. {#parrots}

@bender2021parrots describe a **[large language model](glossary.html#llm)** as a stochastic parrot, a "system for haphazardly stitching together sequences of linguistic forms it has observed in its vast training data, according to probabilistic information about how they combine, but without any reference to meaning." Like parrots mimicking the sounds that they hear around them, LLMs repeat the phrases they are exposed to, but have no conception of the human meaning behind these phrases.

This analogy is useful because it references the very real problem of machine-learning models simply outputting their most frequent training data. Critics of the stochastic-parrot analogy say that it undervalues the competencies that state-of-the-art language models have. Some critics take this further and say that these competencies imply models understand meaning in a human-like way [@piantadosi2022meaning].[^4] For example, proponents of this analogy might argue that Generative AI passing a difficult standardized exam (such as the Bar Exam [@katz2023gpt] or the GRE [@gpt4]) is more about parroting training data than human-like skill.

## Large language models are noisy search engines. {#search}

A search engine allows users to search for information within a large database using natural language queries. Like a search engine, **[large language models](glossary.html#llm)** also return information in response to a natural language query. However, while a search engine queries the entries in its database and returns the most appropriate ones, a language model does not have direct access to its **[training data](glossary.html#datasets)** and can only make predictions based on the information stored in the model **[weights](glossary.html#weights)**.[^5] Most often the output will be a mixture of information contained in many database entries. Some model outputs may quote directly from relevant entries in the database (in the case of **[memorization](glossary.html#memorization)**), but this is not reflective of the most typical outputs.

Sometimes **[generations](glossary.html#generation)** from an LLM will convey similar information that one might learn from running a search; however, sometimes it will not because the underlying **[algorithm](glossary.html#algorithm)** is different. Thus, while some generations answer the **[prompt](glossary.html#prompt)** in a similar way to a search, we can more generally think of generative-model outputs as a noisy version of what is actually in the database. Currently, such outputs also tend to lack attribution to the original data entries, and sometimes are incorrect.

[^1]: Maximizing a reward is exactly equivalent to minimizing a     **[loss](glossary.html#loss)** (except for the extra minus sign), but due to     historical reasons, machine-learning practitioners use the latter     phrasing more often.

[^2]: Curriculum learning is an entire field of research in machine     learning, but it is not currently standard to use a curriculum.

[^3]: § 101 (definition of "compilation").

[^4]: Whether models are human-like, or the outputs are simply "really     good" is less pertinent for how generations and inputs should be     regulated.

[^5]: The training data is seen during training, but models are used     separately from the training data. 

# References