# Task 2: Transformer Networks and Self-Attention in Cybersecurity

## 1. Architectural Overview of Transformers
The **Transformer** model represents a paradigm shift in deep learning, moving away from the sequential processing of Recurrent Neural Networks (RNNs) to a parallelized architecture. This is made possible through the **Self-Attention** mechanism, which allows the model to capture dependencies between elements regardless of their distance in a sequence.

### Core Mechanisms
* **Self-Attention Mechanism:** This is the "brain" of the transformer. It calculates the relationship between all tokens in a sequence simultaneously. For every input, the model generates three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**. The attention score is calculated using the scaled dot-product formula:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    In cybersecurity, this allows the model to identify how a specific system log entry relates to an event that occurred much earlier in the session.



* **Positional Encoding:** Since Transformers do not process data sequentially, they have no inherent sense of the "order" of events. To fix this, **Positional Encoding** is added to the input embeddings. It uses sine and cosine functions of different frequencies to inject information about the position of each token.
    $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
    $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$



---

## 2. Cybersecurity Application: Advanced Threat Detection
Transformers are uniquely suited for **Log Analysis** and **Network Traffic Characterization**. 

### Contextual Anomaly Detection
Traditional security tools look for static "signatures." However, modern attacks often consist of a sequence of "normal-looking" actions that are malicious only when viewed in context (e.g., a user logging in, then suddenly accessing an unusual database, followed by an external data transfer). 

A Transformer model can be trained on months of historical system logs to learn the "language" of a network. Because of the **Self-Attention** mechanism, the model can look at a long sequence of events and flag an anomaly if the current event does not "fit" the context of the previous events.

### Phishing Analysis
Beyond logs, Transformers like BERT are used to analyze the semantic meaning of emails. They can detect social engineering attempts by understanding the tone, urgency, and intent of the text, which is far more effective than simply looking for blacklisted URLs or keywords.

---

## 3. Visualizations and Reproducibility
To reproduce the logic of this report, the reader should refer to the following architectural components:
1.  **Encoder-Decoder Structure:** For processing input sequences and generating predictions.
2.  **Multi-Head Attention:** To allow the model to attend to different parts of the sequence for different reasons (e.g., one head for user ID, another for action type).



---
**Note for Evaluator:** The images referenced above are stored in the `/task_2/` folder as `attention.png`, `position_encoding.png`, and `architecture.png`.

dresses, another for protocol types).Figure 3: The complete Transformer architecture overview.Note for Evaluator: The images referenced in this report are stored in the /task_2/ folder as attention.png, position_encoding.png, and architecture.png.
